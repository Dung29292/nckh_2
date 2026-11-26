"""Retrieval-augmented, text-guided depth-free 3D completion package for nckh_2."""
from __future__ import annotations

from nckh_2.backends.smoothing import (
    bilateral_normal_smooth,
    compute_discrete_curvature,
    interpolate_point_confidence_to_mesh,
    pymeshlab_two_step_smooth,
    transfer_observed_detail_residuals,
)
from nckh_2.evaluation.metrics import (
    DetailRetentionMetrics,
    SurfaceMetrics,
    build_synthetic_detailed_box,
    compute_surface_metrics,
    evaluate_detail_preservation,
    evaluate_reconstruction_split,
)
from nckh_2.pipeline.completion.inpaint_fuse import CompletionResult, run_depth_free_completion
from nckh_2.pipeline.completion.prior_parser import StructuralPrior, parse_structural_prior
from nckh_2.pipeline.completion.visibility import (
    CalibratedView,
    VirtualViewRender,
    VisualHull,
    build_visual_hull,
    load_calibrated_views,
    render_virtual_views,
    seed_symmetry_points,
    solve_biharmonic_depth_proxy,
)
from nckh_2.pipeline.completion.web_retriever import VerifiedReference, retrieve_and_verify_references

__all__ = [
    "CalibratedView",
    "CompletionResult",
    "DetailRetentionMetrics",
    "StructuralPrior",
    "SurfaceMetrics",
    "VerifiedReference",
    "VirtualViewRender",
    "VisualHull",
    "bilateral_normal_smooth",
    "build_synthetic_detailed_box",
    "build_visual_hull",
    "compute_discrete_curvature",
    "compute_surface_metrics",
    "evaluate_detail_preservation",
    "evaluate_reconstruction_split",
    "interpolate_point_confidence_to_mesh",
    "load_calibrated_views",
    "parse_structural_prior",
    "pymeshlab_two_step_smooth",
    "render_virtual_views",
    "retrieve_and_verify_references",
    "run_depth_free_completion",
    "seed_symmetry_points",
    "solve_biharmonic_depth_proxy",
    "transfer_observed_detail_residuals",
]
