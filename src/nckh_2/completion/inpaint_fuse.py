"""Phase 4: Reference-guided depth-free 2D completion & confidence-weighted 3D fusion.

Combines:
  - Stage 1A: Semantic & structural prior extraction (``prior_parser``)
  - Stage 1B: Web reference image retrieval & verification (``web_retriever``)
  - Stage 2 : COLMAP visibility, 3D Visual Hull carving & Biharmonic depth proxy (``visibility``)
  - Stage 3 : Sequential anti-Janus Warp-and-Inpaint / Exact-SKU web blending + metric 3D lifting
              (completely Depth-Anything-free!) + optional ``gsplat`` consolidation.
"""
from __future__ import annotations

import dataclasses
import gc
import json
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.spatial import cKDTree

from .prior_parser import StructuralPrior, parse_structural_prior
from .visibility import (
    VirtualViewRender,
    VisualHull,
    _splat_points_zbuf,
    build_visual_hull,
    estimate_object_frame,
    load_calibrated_views,
    render_virtual_views,
    seed_symmetry_points,
    solve_biharmonic_depth_proxy,
)
from .web_retriever import (
    VerifiedReference,
    retrieve_and_verify_references,
    warp_reference_to_view,
)


@dataclasses.dataclass
class CompletionResult:
    """Output bundle from depth-free retrieval-augmented 3D completion."""

    point_cloud: o3d.geometry.PointCloud
    confidence: np.ndarray              # (N,) per-point confidence in [0, 1] (1.0 = observed MVS)
    observed_points: np.ndarray         # (N_obs, 3) immutable observed points for detail baking
    prior: StructuralPrior
    verified_refs: dict[str, VerifiedReference]
    stats: dict[str, Any]


# ---------------------------------------------------------------------------
# COLOR HARMONIZATION & DUAL-PATH 2D COMPLETION
# ---------------------------------------------------------------------------

def harmonize_lab_color(
    src_bgr: np.ndarray,
    src_mask: np.ndarray,
    target_lab_stats: dict[str, list[float]],
    blend_alpha: float = 0.70,
) -> np.ndarray:
    """Reinhard CIE Lab mean/std color transfer so studio web photos match real capture lighting."""
    fg = src_mask > 127
    if np.sum(fg) < 30:
        return src_bgr.copy()

    lab = cv2.cvtColor(src_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    t_mean = np.asarray(target_lab_stats.get("mean", [128.0, 128.0, 128.0]), dtype=np.float32)
    t_std = np.asarray(target_lab_stats.get("std", [35.0, 15.0, 15.0]), dtype=np.float32)

    s_mean = lab[fg].mean(axis=0)
    s_std = np.maximum(lab[fg].std(axis=0), 5.0)
    # Limit extreme scaling to avoid amplifying JPEG compression noise
    scale = np.clip(t_std / s_std, 0.55, 1.65)

    shifted = (lab[fg] - s_mean[None, :]) * scale[None, :] + t_mean[None, :]
    lab_out = lab.copy()
    lab_out[fg] = np.clip((1.0 - blend_alpha) * lab[fg] + blend_alpha * shifted, 0.0, 255.0)
    return cv2.cvtColor(lab_out.astype(np.uint8), cv2.COLOR_LAB2BGR)


def _feather_blend_seam(
    base_bgr: np.ndarray,
    fill_bgr: np.ndarray,
    mask_occ: np.ndarray,
    mask_obs: np.ndarray,
    feather_px: int = 10,
) -> np.ndarray:
    """Blend completed pixels inside ``mask_occ`` smoothly across boundary seams with ``mask_obs``."""
    out = base_bgr.copy()
    if not np.any(mask_occ):
        return out

    # Distance inside mask_occ from observed pixels -> smooth alpha ramp from 0..1 across feather_px
    dist_in = ndimage.distance_transform_edt(mask_occ)
    alpha = np.clip(dist_in / float(max(feather_px, 1)), 0.0, 1.0).astype(np.float32)

    # Dilate base colors outward slightly to give smooth boundary reference for seam pixels
    obs_valid = mask_obs & np.any(base_bgr > 4, axis=2)
    if np.any(obs_valid):
        _, idx = ndimage.distance_transform_edt(~obs_valid, return_indices=True)
        base_ext = base_bgr[idx[0], idx[1]]
    else:
        base_ext = fill_bgr

    blended = (alpha[:, :, None] * fill_bgr.astype(np.float32) + (1.0 - alpha[:, :, None]) * base_ext.astype(np.float32))
    out[mask_occ] = np.clip(blended[mask_occ], 0, 255).astype(np.uint8)
    return out


def _fast_structure_guided_inpaint(
    view: VirtualViewRender,
    prior: StructuralPrior,
) -> np.ndarray:
    """Fast 0-VRAM boundary + horizontal color-band guided inpainting inside ``view.mask_occ``."""
    h, w = view.height, view.width
    rgb = view.rgb_obs.copy()
    occ = view.mask_occ
    obs_valid = view.mask_obs & np.any(rgb > 4, axis=2)

    if not np.any(obs_valid):
        # Fully unseen view without observed border: synthesize from observed Lab mean color
        mean_lab = np.uint8([[np.clip(prior.observed_lab_stats["mean"], 0, 255)]])
        fill_bgr = cv2.cvtColor(mean_lab, cv2.COLOR_LAB2BGR)[0, 0]
        rgb[occ] = fill_bgr
        return rgb

    # Fill row-stratified median colors (preserves horizontal lid/body bands on boxes & bottles)
    fill_canvas = rgb.copy()
    global_med = np.median(rgb[obs_valid], axis=0).astype(np.uint8)
    for y in range(h):
        if not np.any(occ[y]):
            continue
        obs_row = obs_valid[y]
        if np.sum(obs_row) >= 4:
            row_col = np.median(rgb[y, obs_row], axis=0).astype(np.uint8)
        else:
            row_col = global_med
        fill_canvas[y, occ[y]] = row_col

    # Refine near observed boundaries via Telea inpainting so local edges & shadows continue smoothly
    inpaint_mask = (occ.astype(np.uint8)) * 255
    telea = cv2.inpaint(rgb, inpaint_mask, inpaintRadius=7, flags=cv2.INPAINT_TELEA)
    dist = ndimage.distance_transform_edt(occ)
    w_telea = np.exp(-dist / 24.0).astype(np.float32)
    combined = w_telea[:, :, None] * telea.astype(np.float32) + (1.0 - w_telea[:, :, None]) * fill_canvas.astype(np.float32)
    out = rgb.copy()
    out[occ] = np.clip(combined[occ], 0, 255).astype(np.uint8)
    return out


def _try_diffusers_controlnet_inpaint(
    view: VirtualViewRender,
    prompt_text: str,
    ref_bgr: np.ndarray | None = None,
) -> np.ndarray | None:
    """Optional local SD 1.5 Inpainting + ControlNet-Depth (fed with Biharmonic D_proxy, fp16)."""
    try:
        import torch
        from diffusers import ControlNetModel, StableDiffusionControlNetInpaintPipeline
        from PIL import Image
    except ImportError:
        return None

    if not torch.cuda.is_available() or view.depth_proxy is None:
        return None

    device = "cuda"
    pipe = None
    try:
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/sd-controlnet-depth",
            torch_dtype=torch.float16,
        )
        pipe = StableDiffusionControlNetInpaintPipeline.from_pretrained(
            "runwayml/stable-diffusion-inpainting",
            controlnet=controlnet,
            torch_dtype=torch.float16,
            safety_checker=None,
        )
        pipe.enable_attention_slicing()
        pipe.enable_model_cpu_offload()

        # Normalize metric Biharmonic depth proxy into [0, 255] disparity map for ControlNet-Depth
        d = view.depth_proxy.copy()
        valid = view.mask_vh & (d > 1e-4)
        disp = np.zeros_like(d, dtype=np.uint8)
        if np.any(valid):
            d_min, d_max = float(np.percentile(d[valid], 2)), float(np.percentile(d[valid], 98))
            norm = 1.0 - np.clip((d - d_min) / max(d_max - d_min, 1e-4), 0.0, 1.0)
            disp[valid] = (norm[valid] * 255.0).astype(np.uint8)

        sz = 512
        init_rgb = cv2.cvtColor(cv2.resize(view.rgb_obs, (sz, sz)), cv2.COLOR_BGR2RGB)
        msk_u8 = cv2.resize((view.mask_occ.astype(np.uint8) * 255), (sz, sz), interpolation=cv2.INTER_NEAREST)
        ctrl_rgb = cv2.cvtColor(cv2.resize(disp, (sz, sz)), cv2.COLOR_GRAY2RGB)

        result = pipe(
            prompt=prompt_text,
            negative_prompt="blurry, distorted, extra parts, watermark, text overlay, artifacts",
            image=Image.fromarray(init_rgb),
            mask_image=Image.fromarray(msk_u8),
            control_image=Image.fromarray(ctrl_rgb),
            num_inference_steps=20,
            guidance_scale=7.0,
            controlnet_conditioning_scale=0.85,
        ).images[0]
        res_bgr = cv2.cvtColor(np.array(result), cv2.COLOR_RGB2BGR)
        return cv2.resize(res_bgr, (view.width, view.height), interpolation=cv2.INTER_LINEAR)
    except Exception as exc:
        print(f"  [Inpaint] Diffusers ControlNet fallback ({exc})")
        return None
    finally:
        if pipe is not None:
            del pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def complete_virtual_view_2d(
    view: VirtualViewRender,
    prior: StructuralPrior,
    ref: VerifiedReference | None = None,
    inpaint_backend: str = "fast_guided",
) -> tuple[np.ndarray, float, str]:
    """Complete occluded 2D pixels of ``view`` using Path 2 (web reference) or Path 1 (inpainting).

    Returns ``(completed_bgr, max_confidence, method_tag)``.
    """
    if not np.any(view.mask_occ):
        return view.rgb_obs.copy(), 1.0, "fully_observed"

    # Path 2: Verified Web Reference Image directly aligned onto the Visual Hull silhouette
    if ref is not None and ref.cutout_path.exists() and ref.mask_path.exists():
        ref_cut = cv2.imread(str(ref.cutout_path))
        ref_msk = cv2.imread(str(ref.mask_path), cv2.IMREAD_GRAYSCALE)
        if ref_cut is not None and ref_msk is not None:
            harm_cut = harmonize_lab_color(ref_cut, ref_msk, prior.observed_lab_stats, blend_alpha=0.68)
            warped_bgr, warped_msk = warp_reference_to_view(harm_cut, ref_msk, view.mask_vh, flip_x=ref.flip_x)
            # Fill any uncovered sub-pixels inside mask_occ with fast boundary fallback first
            fallback = _fast_structure_guided_inpaint(view, prior)
            merged_fill = fallback.copy()
            valid_ref = (warped_msk > 127) & view.mask_occ
            merged_fill[valid_ref] = warped_bgr[valid_ref]
            blended = _feather_blend_seam(view.rgb_obs, merged_fill, view.mask_occ, view.mask_obs, feather_px=10)
            conf = 0.78 if (ref.verify_score >= 0.75 and ref.silhouette_iou >= 0.75) else 0.68
            return blended, conf, f"web_ref({ref.ref_id}, sim={ref.verify_score:.2f}, iou={ref.silhouette_iou:.2f})"

    # Path 1A: Optional local Diffusers ControlNet-Depth fed with metric Biharmonic D_proxy
    if inpaint_backend == "diffusers":
        p_txt = prior.view_prompts.get(view.name, prior.prompt)
        diff_bgr = _try_diffusers_controlnet_inpaint(view, p_txt)
        if diff_bgr is not None:
            blended = _feather_blend_seam(view.rgb_obs, diff_bgr, view.mask_occ, view.mask_obs, feather_px=10)
            return blended, 0.62, "diffusers_controlnet_proxy"

    # Path 1B: Fast 0-VRAM structure & color-band guided inpainting
    fast_bgr = _fast_structure_guided_inpaint(view, prior)
    blended = _feather_blend_seam(view.rgb_obs, fast_bgr, view.mask_occ, view.mask_obs, feather_px=8)
    return blended, 0.58, "fast_structure_guided"


# ---------------------------------------------------------------------------
# OPTIONAL GSPLAT MULTI-VIEW PHOTOMETRIC CONSOLIDATION (DEPTH-FREE RELIEF)
# ---------------------------------------------------------------------------

def _try_gsplat_consolidate(
    pts: np.ndarray,
    cols_rgb: np.ndarray,
    conf: np.ndarray,
    completed_views: list[tuple[VirtualViewRender, np.ndarray]],
    iters: int = 120,
) -> tuple[np.ndarray, np.ndarray]:
    """Optional lightweight gsplat color & position consolidation across overlapping virtual views.

    Observed points (conf >= 0.99) stay frozen while synthesized points adjust smoothly
    to satisfy multi-view photometric consistency across overlapping virtual cameras.
    """
    try:
        import torch
        from gsplat import rasterization
    except Exception:
        return pts, cols_rgb

    if not torch.cuda.is_available() or len(completed_views) < 2 or len(pts) < 100:
        return pts, cols_rgb

    device = torch.device("cuda")
    try:
        diag = float(np.linalg.norm(np.percentile(pts, 95, axis=0) - np.percentile(pts, 5, axis=0)))
        init_scale = max(0.008 * diag, 1e-4)

        is_syn = conf < 0.95
        means_obs = torch.tensor(pts[~is_syn], dtype=torch.float32, device=device)
        cols_obs = torch.tensor(np.clip(cols_rgb[~is_syn], 1e-3, 1.0 - 1e-3), dtype=torch.float32, device=device)

        means_syn = torch.nn.Parameter(torch.tensor(pts[is_syn], dtype=torch.float32, device=device))
        cols_syn = torch.nn.Parameter(torch.tensor(np.clip(cols_rgb[is_syn], 1e-3, 1.0 - 1e-3), dtype=torch.float32, device=device))
        means_syn_anchor = means_syn.detach().clone()

        n_all = len(pts)
        quats = torch.zeros((n_all, 4), dtype=torch.float32, device=device)
        quats[:, 0] = 1.0
        scales = torch.full((n_all, 3), init_scale, dtype=torch.float32, device=device)
        opacities = torch.full((n_all,), 0.85, dtype=torch.float32, device=device)

        opt = torch.optim.Adam([
            {"params": [means_syn], "lr": 0.002 * diag},
            {"params": [cols_syn], "lr": 0.02},
        ])

        cam_tensors = []
        for vw, comp_bgr in completed_views:
            viewmat = np.eye(4, dtype=np.float32)
            viewmat[:3, :3] = vw.R.astype(np.float32)
            viewmat[:3, 3] = vw.t.astype(np.float32)
            tgt = cv2.cvtColor(comp_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            msk = vw.mask_occ.astype(np.float32)
            cam_tensors.append((
                torch.tensor(viewmat, device=device)[None],
                torch.tensor(vw.K.astype(np.float32), device=device)[None],
                vw.width,
                vw.height,
                torch.tensor(tgt, device=device),
                torch.tensor(msk, device=device)[..., None],
            ))

        for step in range(max(1, int(iters))):
            vm, K_t, w, h, tgt, msk = cam_tensors[step % len(cam_tensors)]
            means = torch.cat([means_obs, means_syn], dim=0)
            colors = torch.clamp(torch.cat([cols_obs, cols_syn], dim=0), 0.0, 1.0)
            renders, _, _ = rasterization(
                means=means, quats=quats, scales=scales, opacities=opacities,
                colors=colors, viewmats=vm, Ks=K_t, width=w, height=h,
            )
            pred = renders[0]
            loss_rgb = torch.mean(torch.abs(pred - tgt) * msk)
            loss_reg = 0.05 * torch.mean((means_syn - means_syn_anchor) ** 2) / max(diag * diag, 1e-6)
            loss = loss_rgb + loss_reg
            opt.zero_grad()
            loss.backward()
            opt.step()

        out_pts = pts.copy()
        out_cols = cols_rgb.copy()
        out_pts[is_syn] = means_syn.detach().cpu().numpy().astype(np.float64)
        out_cols[is_syn] = np.clip(cols_syn.detach().cpu().numpy(), 0.0, 1.0).astype(np.float64)
        return out_pts, out_cols
    except Exception as exc:
        print(f"  [gsplat] Bo qua consolidation ({exc})")
        return pts, cols_rgb
    finally:
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# MAIN ORCHESTRATOR: DEPTH-FREE RETRIEVAL-AUGMENTED 3D COMPLETION
# ---------------------------------------------------------------------------

def run_depth_free_completion(
    sample_name: str,
    sample_dir: Path,
    pc_raw: o3d.geometry.PointCloud,
    cfg: dict,
) -> CompletionResult:
    """Execute the full Depth-Free Retrieval-Augmented 3D Completion pipeline on ``pc_raw``.

    Preserves all observed MVS points at ``confidence = 1.0`` and fills only occluded regions
    inside the calibrated Step 2 Visual Hull using symmetry seeding + web reference retrieval +
    Biharmonic metric surface lifting.
    """
    t0 = time.time()
    c_cfg = cfg.get("completion", {})
    prompt = str(cfg.get("masking", {}).get("prompt", "object"))

    # 0. Clean floating COLMAP outliers via SOR + DBSCAN (keep multi-part clusters >= 3% of largest)
    pc_obs, _ = pc_raw.remove_statistical_outlier(nb_neighbors=25, std_ratio=1.8)
    raw_pts_tmp = np.asarray(pc_obs.points, dtype=np.float64)
    if len(raw_pts_tmp) > 80:
        diag_tmp = float(np.linalg.norm(np.percentile(raw_pts_tmp, 95, axis=0) - np.percentile(raw_pts_tmp, 5, axis=0)))
        eps = max(0.025 * diag_tmp, 1e-3)
        labels = np.array(pc_obs.cluster_dbscan(eps=eps, min_points=35, print_progress=False))
        if len(labels) > 0 and np.any(labels >= 0):
            counts = np.bincount(labels[labels >= 0])
            max_c = int(counts.max())
            keep_ids = np.where(counts >= max(35, int(0.03 * max_c)))[0]
            keep_idx = np.nonzero(np.isin(labels, keep_ids))[0]
            if len(keep_idx) >= 50:
                pc_obs = pc_obs.select_by_index(keep_idx)

    obs_pts = np.asarray(pc_obs.points, dtype=np.float64)
    obs_cols = (
        np.asarray(pc_obs.colors, dtype=np.float64)
        if pc_obs.has_colors()
        else np.full_like(obs_pts, 0.4)
    )
    center, _, extents = estimate_object_frame(obs_pts)
    diag = float(np.linalg.norm(extents))
    min_gap = float(c_cfg.get("lift_spacing_frac", 0.012)) * max(diag, 0.1)

    # 1A. Semantic & structural prior extraction
    refs_dir = sample_dir / "web_refs"
    prior = parse_structural_prior(
        prompt=prompt,
        sample_name=sample_name,
        sample_dir=sample_dir,
        backend=str(c_cfg.get("prior_backend", "auto")),
        cache_path=refs_dir / "structural_prior.json",
    )
    print(f"  [Completion 1A] Category='{prior.object_category}' | Colors={prior.observed_color_names} | Sym={prior.bilateral_axes}/{prior.rotational_order}")

    # 2. Calibrated COLMAP views & 3D Visual Hull carving
    views = load_calibrated_views(sample_dir)
    vh = build_visual_hull(
        obs_pts,
        views,
        grid_res=int(c_cfg.get("visual_hull_res", 64)),
        pad_frac=float(c_cfg.get("visual_hull_pad_frac", 0.28)),
        min_mask_consistency=float(c_cfg.get("visual_hull_consistency", 0.80)),
    )
    if vh is not None:
        vh_surf, _ = vh.extract_surface_points()
        print(f"  [Completion 2 ] Visual Hull carved from {len(views)} views ({len(vh_surf):,} envelope voxels)")
    else:
        print(f"  [Completion 2 ] Visual Hull skipped (<2 calibrated masks); using 2D silhouette hull fallback")

    # 3A. ICP-verified symmetry seeding (strictly inside gaps & Visual Hull)
    sym_pts, sym_cols, sym_conf = seed_symmetry_points(
        obs_pts,
        obs_cols,
        bilateral_axes=prior.bilateral_axes if c_cfg.get("use_symmetry", True) else [],
        rotational_order=prior.rotational_order if c_cfg.get("use_symmetry", True) else 0,
        visual_hull=vh,
        min_overlap=float(c_cfg.get("symmetry_min_overlap", 0.72)),
        gap_frac=float(c_cfg.get("symmetry_gap_frac", 0.025)),
    )
    if len(sym_pts) > 0:
        print(f"  [Completion 3A] ICP-verified symmetry injected {len(sym_pts):,} points (w=0.75)")

    acc_pts = np.vstack([obs_pts, sym_pts]) if len(sym_pts) > 0 else obs_pts.copy()
    acc_cols = np.vstack([obs_cols, sym_cols]) if len(sym_pts) > 0 else obs_cols.copy()
    acc_conf = np.concatenate([np.ones(len(obs_pts)), sym_conf]) if len(sym_pts) > 0 else np.ones(len(obs_pts))

    # Render initial virtual views to discover remaining occluded faces
    v_size = int(c_cfg.get("virtual_view_size", 320))
    virtual_views = render_virtual_views(acc_pts, acc_cols, visual_hull=vh, image_size=v_size)

    # 1B. Prompt-driven web image research, verification & alignment
    verified_refs: dict[str, VerifiedReference] = {}
    if c_cfg.get("enable_web_retrieval", True):
        verified_refs = retrieve_and_verify_references(
            prior=prior,
            sample_dir=sample_dir,
            virtual_views=virtual_views,
            provider=str(c_cfg.get("web_provider", "duckduckgo")),
            max_per_query=int(c_cfg.get("web_max_per_query", 5)),
            min_verify_score=float(c_cfg.get("web_min_verify_score", 0.56)),
            enable_online=bool(c_cfg.get("web_online", True)),
        )
        print(f"  [Completion 1B] Verified web/manual references matched to {len(verified_refs)} occluded views: {list(verified_refs.keys())}")

    # 3B & 3C. Sequential Anti-Janus Warp-and-Inpaint + Metric Biharmonic 3D Lifting
    # Sort occluded views by highest observed boundary context first (most anchors -> least drift)
    occluded_views = [v for v in virtual_views if v.occlusion_ratio >= float(c_cfg.get("min_occlusion_ratio", 0.06))]
    occluded_views.sort(key=lambda v: v.boundary_context_ratio, reverse=True)

    debug_dir = refs_dir / "virtual_views"
    debug_dir.mkdir(parents=True, exist_ok=True)

    obs_tree = cKDTree(obs_pts)
    sigma_d = max(0.35 * diag, 1e-3)
    inpaint_backend = str(c_cfg.get("inpaint_backend", "fast_guided"))
    stride = max(1, int(c_cfg.get("lift_pixel_stride", 2)))
    completed_view_pairs: list[tuple[VirtualViewRender, np.ndarray]] = []
    lifted_total = 0

    for vw in occluded_views:
        # Forward-warp latest accumulated 3D points into this virtual view (prevents Janus double-synthesis!)
        acc_bgr = np.clip(acc_cols[:, ::-1] * 255.0, 0, 255).astype(np.uint8)
        rgb_cur, depth_cur, mask_cur = _splat_points_zbuf(
            acc_pts, acc_bgr, vw.K, vw.R, vw.t, vw.width, vw.height, splat_radius=2
        )
        mask_cur_closed = ndimage.binary_closing(mask_cur, iterations=2)
        vw.rgb_obs = rgb_cur
        vw.depth_obs = depth_cur
        vw.mask_obs = mask_cur_closed
        vw.mask_occ = vw.mask_vh & (~mask_cur_closed)

        if vw.occlusion_ratio < 0.03:
            continue

        # Solve exact metric Biharmonic depth membrane (zero Depth Anything!)
        d_proxy = solve_biharmonic_depth_proxy(vw, vh_prior_weight=float(c_cfg.get("vh_prior_weight", 0.08)))

        # Complete occluded 2D pixels (Path 2 Web Ref OR Path 1 Inpainting)
        comp_bgr, w_max, method_tag = complete_virtual_view_2d(
            vw, prior, ref=verified_refs.get(vw.name), inpaint_backend=inpaint_backend
        )
        completed_view_pairs.append((vw, comp_bgr))

        # Save side-by-side debug visualization [Observed | Occlusion Mask | Completed]
        occ_vis = cv2.cvtColor((vw.mask_occ.astype(np.uint8) * 255), cv2.COLOR_GRAY2BGR)
        panel = np.hstack([vw.rgb_obs, occ_vis, comp_bgr])
        cv2.imwrite(str(debug_dir / f"{vw.name}_completion.jpg"), panel, [cv2.IMWRITE_JPEG_QUALITY, 90])

        # Back-project occluded pixels onto metric Biharmonic surface D_proxy (subsampled by stride)
        lift_mask = np.zeros_like(vw.mask_occ)
        lift_mask[::stride, ::stride] = vw.mask_occ[::stride, ::stride]
        new_pts = vw.unproject_pixels(lift_mask, d_proxy)
        if len(new_pts) == 0:
            continue

        ys, xs = np.nonzero(lift_mask & (d_proxy > 1e-4))
        new_cols = (comp_bgr[ys, xs, ::-1].astype(np.float64) / 255.0)

        # Clip strictly to Step 2 Visual Hull + avoid duplicating already-populated geometry
        keep = np.ones(len(new_pts), dtype=bool)
        if vh is not None:
            keep &= vh.contains_points(new_pts, dilate_voxels=1)
        acc_tree = cKDTree(acc_pts)
        d_acc, _ = acc_tree.query(new_pts, k=1)
        keep &= d_acc > min_gap

        if np.any(keep):
            sel_pts = new_pts[keep]
            sel_cols = new_cols[keep]
            d_obs, _ = obs_tree.query(sel_pts, k=1)
            sel_conf = np.clip(w_max * np.exp(-d_obs / sigma_d), 0.25, w_max)

            acc_pts = np.vstack([acc_pts, sel_pts])
            acc_cols = np.vstack([acc_cols, sel_cols])
            acc_conf = np.concatenate([acc_conf, sel_conf])
            lifted_total += len(sel_pts)
            print(f"  [Completion 3B] View '{vw.name:10s}' ({method_tag}): +{len(sel_pts):,} pts (occ={vw.occlusion_ratio:.0%})")

    # Optional gsplat multi-view consolidation for higher photometric & parallax relief fidelity
    if bool(c_cfg.get("use_gsplat_consolidation", False)) and lifted_total > 0 and len(completed_view_pairs) >= 2:
        print("  [Completion 3C] Running optional gsplat multi-view consolidation...")
        acc_pts, acc_cols = _try_gsplat_consolidate(
            acc_pts, acc_cols, acc_conf, completed_view_pairs,
            iters=int(c_cfg.get("gsplat_iters", 120)),
        )

    # Build final Open3D PointCloud with consistent outward normals
    pc_out = o3d.geometry.PointCloud()
    pc_out.points = o3d.utility.Vector3dVector(acc_pts)
    pc_out.colors = o3d.utility.Vector3dVector(np.clip(acc_cols, 0.0, 1.0))
    pc_out.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=max(0.06 * diag, 0.01), max_nn=35)
    )
    # Orient normals consistently outward from object centroid
    norms = np.asarray(pc_out.normals)
    rad_vec = acc_pts - center[None, :]
    flip = np.sum(norms * rad_vec, axis=1) < 0
    norms[flip] *= -1.0
    pc_out.normals = o3d.utility.Vector3dVector(norms)

    dur = time.time() - t0
    stats = {
        "observed_points": int(len(obs_pts)),
        "symmetry_points": int(len(sym_pts)),
        "lifted_points": int(lifted_total),
        "total_points": int(len(acc_pts)),
        "verified_web_views": list(verified_refs.keys()),
        "completed_virtual_views": [v.name for v, _ in completed_view_pairs],
        "elapsed_sec": round(dur, 2),
    }
    (refs_dir / "completion_report.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(
        f"  [Completion] Done in {dur:.2f}s: {len(obs_pts):,} observed + {len(sym_pts):,} sym "
        f"+ {lifted_total:,} lifted = {len(acc_pts):,} pts"
    )
    return CompletionResult(
        point_cloud=pc_out,
        confidence=acc_conf,
        observed_points=obs_pts,
        prior=prior,
        verified_refs=verified_refs,
        stats=stats,
    )
