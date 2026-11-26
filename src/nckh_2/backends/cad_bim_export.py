"""Analytical 2D CAD (.dxf) & 3D BIM IFC4 (.ifc) Exporters + Pre-Export Validation Gates (Section 5).

Maps extracted geometric entities directly to industry CAD/BIM formats (Table 1 of nckh.pdf):
  - Architectural Planes (Walls, Floors, Ceilings) -> 2D CAD `.dxf` LWPOLYLINE/LINE & BIM `IfcWall`, `IfcSlab`
  - Analytical Quadrics (Circular Columns)         -> 2D CAD `.dxf` CIRCLE/ARC & BIM `IfcColumn`
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from nckh_2.backends.ransac_manhattan import WallSegment2D
from nckh_2.backends.surfaces import CircularCylinder


def validate_room_polygon_for_export(
    corners_2d: np.ndarray,
    export_cfg: dict[str, Any],
) -> dict[str, Any]:
    """Verify closed loop topology and 90-degree Manhattan corner accuracy before CAD/BIM export."""
    pts = np.asarray(corners_2d, dtype=np.float64)
    if len(pts) < 3:
        return {"valid": False, "reason": "fewer_than_3_vertices", "num_vertices": len(pts)}

    # Check non-zero signed shoelace area
    x, y = pts[:, 0], pts[:, 1]
    signed_area = 0.5 * float(np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y))
    area = abs(signed_area)

    ortho_tol = float(export_cfg["corner_orthogonality_tol_deg"])
    angle_devs: list[float] = []
    n_v = len(pts)
    for i in range(n_v):
        v_prev = pts[(i - 1) % n_v] - pts[i]
        v_next = pts[(i + 1) % n_v] - pts[i]
        l1, l2 = float(np.linalg.norm(v_prev)), float(np.linalg.norm(v_next))
        if l1 > 1.0e-9 and l2 > 1.0e-9:
            cos_a = float(np.clip(np.dot(v_prev, v_next) / (l1 * l2), -1.0, 1.0))
            ang_deg = float(np.degrees(np.arccos(cos_a)))
            angle_devs.append(abs(ang_deg - 90.0))

    max_ortho_err = max(angle_devs) if angle_devs else 0.0
    return {
        "valid": area > 1.0e-4,
        "floor_area_m2": round(area, 6),
        "num_corners": n_v,
        "max_90deg_error_deg": round(max_ortho_err, 6),
        "is_strictly_manhattan": max_ortho_err <= ortho_tol,
    }


def export_floorplan_dxf(
    out_path: Path | str,
    wall_segments: list[WallSegment2D],
    room_polygon_2d: np.ndarray | None,
    columns: list[CircularCylinder],
    export_cfg: dict[str, Any],
) -> Path:
    """Export analytical 2D CAD floorplan (.dxf AC1009/R12 ASCII) with LINE and CIRCLE entities."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    wall_layer = str(export_cfg.get("dxf_wall_layer", export_cfg.get("dxf_Wall_layer", "A-WALL")))
    col_layer = str(export_cfg.get("dxf_column_layer", "A-COLS"))
    slab_layer = str(export_cfg.get("dxf_slab_layer", "A-SLAB"))

    lines: list[str] = [
        "0", "SECTION",
        "2", "HEADER",
        "9", "$ACADVER", "1", "AC1009",
        "9", "$INSUNITS", "70", "6",  # 6 = Meters
        "0", "ENDSEC",
        "0", "SECTION",
        "2", "ENTITIES",
    ]

    # 1. Wall segments (LINE entities)
    for seg in wall_segments:
        lines.extend([
            "0", "LINE",
            "8", wall_layer,
            "10", f"{seg.p_start[0]:.6f}",
            "20", f"{seg.p_start[1]:.6f}",
            "30", "0.0",
            "11", f"{seg.p_end[0]:.6f}",
            "21", f"{seg.p_end[1]:.6f}",
            "31", "0.0",
        ])

    # 2. Closed slab boundary polyline (POLYLINE + VERTEX + SEQEND)
    if room_polygon_2d is not None and len(room_polygon_2d) >= 3:
        poly = np.asarray(room_polygon_2d, dtype=np.float64)
        lines.extend(["0", "POLYLINE", "8", slab_layer, "66", "1", "70", "1"])
        for vx, vy in poly:
            lines.extend(["0", "VERTEX", "8", slab_layer, "10", f"{vx:.6f}", "20", f"{vy:.6f}", "30", "0.0"])
        lines.extend(["0", "SEQEND"])

    # 3. Analytical circular columns (CIRCLE entities from Eq. 4)
    for col in columns:
        lines.extend([
            "0", "CIRCLE",
            "8", col_layer,
            "10", f"{col.anchor[0]:.6f}",
            "20", f"{col.anchor[1]:.6f}",
            "30", "0.0",
            "40", f"{col.radius:.6f}",
        ])

    lines.extend(["0", "ENDSEC", "0", "EOF"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def export_bim_ifc(
    out_path: Path | str,
    wall_segments: list[WallSegment2D],
    room_polygon_2d: np.ndarray | None,
    columns: list[CircularCylinder],
    z_floor: float,
    z_ceiling: float,
    topdown_cfg: dict[str, Any],
    export_cfg: dict[str, Any],
) -> Path:
    """Export analytical IFC4 STEP file containing IfcWall, IfcSlab, and IfcColumn extrusions."""
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    schema = str(export_cfg.get("ifc_schema", "IFC4"))
    proj_name = str(export_cfg.get("ifc_project_name", "NCKH2_ScanToBIM"))
    wall_thick = float(topdown_cfg["default_wall_thickness_m"])
    slab_thick = float(topdown_cfg["default_slab_thickness_m"])
    height = max(float(z_ceiling - z_floor), 0.10)

    entities: list[str] = []
    next_id = 1

    def _add(expr: str) -> int:
        nonlocal next_id
        eid = next_id
        next_id += 1
        entities.append(f"#{eid}={expr};")
        return eid

    origin_3d = _add("IFCCARTESIANPOINT((0.,0.,0.))")
    dir_z = _add("IFCDIRECTION((0.,0.,1.))")
    dir_x = _add("IFCDIRECTION((1.,0.,0.))")
    world_axis = _add(f"IFCAXIS2PLACEMENT3D(#{origin_3d},#{dir_z},#{dir_x})")
    ctx = _add(f"IFCGEOMETRICREPRESENTATIONCONTEXT($,'Model',3,1.E-05,#{world_axis},$)")
    unit_len = _add("IFCSIUNIT(*,.LENGTHUNIT.,$,.METRE.)")
    unit_area = _add("IFCSIUNIT(*,.AREAUNIT.,$,.SQUARE_METRE.)")
    unit_vol = _add("IFCSIUNIT(*,.VOLUMEUNIT.,$,.CUBIC_METRE.)")
    units = _add(f"IFCUNITASSIGNMENT((#{unit_len},#{unit_area},#{unit_vol}))")

    project = _add(f"IFCPROJECT('0NCKH2PRJ0000000000001',$,'{proj_name}',$,$,$,$,(#{ctx}),#{units})")
    site_pl = _add(f"IFCLOCALPLACEMENT($,#{world_axis})")
    site = _add(f"IFCSITE('0NCKH2SIT0000000000002',$,'DefaultSite',$,$,#{site_pl},$,$,.ELEMENT.,$,$,$,$,$)")
    bldg_pl = _add(f"IFCLOCALPLACEMENT(#{site_pl},#{world_axis})")
    bldg = _add(f"IFCBUILDING('0NCKH2BLD0000000000003',$,'DefaultBuilding',$,$,#{bldg_pl},$,$,.ELEMENT.,$,$,$)")
    stry_pt = _add(f"IFCCARTESIANPOINT((0.,0.,{float(z_floor):.6f}))")
    stry_ax = _add(f"IFCAXIS2PLACEMENT3D(#{stry_pt},#{dir_z},#{dir_x})")
    stry_pl = _add(f"IFCLOCALPLACEMENT(#{bldg_pl},#{stry_ax})")
    storey = _add(f"IFCBUILDINGSTOREY('0NCKH2STR0000000000004',$,'Level_0',$,$,#{stry_pl},$,$,.ELEMENT.,{float(z_floor):.6f})")

    _add(f"IFCRELAGGREGATES('0NCKH2REL0000000000005',$,$,$,#{project},(#{site}))")
    _add(f"IFCRELAGGREGATES('0NCKH2REL0000000000006',$,$,$,#{site},(#{bldg}))")
    _add(f"IFCRELAGGREGATES('0NCKH2REL0000000000007',$,$,$,#{bldg},(#{storey}))")

    product_ids: list[int] = []
    origin_2d = _add("IFCCARTESIANPOINT((0.,0.))")
    dir_2d_x = _add("IFCDIRECTION((1.,0.))")
    ax_2d = _add(f"IFCAXIS2PLACEMENT2D(#{origin_2d},#{dir_2d_x})")

    # 1. IfcWall extrusions
    for idx_w, seg in enumerate(wall_segments):
        seg_vec = seg.p_end - seg.p_start
        seg_len = float(np.linalg.norm(seg_vec))
        if seg_len <= 1.0e-4:
            continue
        tan_dir = seg_vec / seg_len
        mid_2d = 0.5 * (seg.p_start + seg.p_end)

        w_pt = _add(f"IFCCARTESIANPOINT(({mid_2d[0]:.6f},{mid_2d[1]:.6f},0.))")
        w_dir = _add(f"IFCDIRECTION(({tan_dir[0]:.6f},{tan_dir[1]:.6f},0.))")
        w_ax = _add(f"IFCAXIS2PLACEMENT3D(#{w_pt},#{dir_z},#{w_dir})")
        w_pl = _add(f"IFCLOCALPLACEMENT(#{stry_pl},#{w_ax})")

        prof = _add(f"IFCRECTANGLEPROFILEDEF(.AREA.,'WallProfile',#{ax_2d},{seg_len:.6f},{wall_thick:.6f})")
        solid = _add(f"IFCEXTRUDEDAREASOLID(#{prof},#{world_axis},#{dir_z},{height:.6f})")
        shape = _add(f"IFCSHAPEREPRESENTATION(#{ctx},'Body','SweptSolid',(#{solid}))")
        pdef = _add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{shape}))")
        wall_id = _add(
            f"IFCWALL('0NCKH2WAL{idx_w:013d}',$,'Wall_{idx_w}',$,$,#{w_pl},#{pdef},$,.STANDARD.)"
        )
        product_ids.append(wall_id)

    # 2. IfcSlab floor extrusion
    if room_polygon_2d is not None and len(room_polygon_2d) >= 3:
        poly = np.asarray(room_polygon_2d, dtype=np.float64)
        pt_refs: list[str] = []
        for vx, vy in poly:
            pid = _add(f"IFCCARTESIANPOINT(({vx:.6f},{vy:.6f}))")
            pt_refs.append(f"#{pid}")
        pt_refs.append(pt_refs[0])
        pline = _add(f"IFCPOLYLINE(({','.join(pt_refs)}))")
        arb_prof = _add(f"IFCARBITRARYCLOSEDPROFILEDEF(.AREA.,'FloorSlabProfile',#{pline})")
        slab_pt = _add(f"IFCCARTESIANPOINT((0.,0.,{-slab_thick:.6f}))")
        slab_ax = _add(f"IFCAXIS2PLACEMENT3D(#{slab_pt},#{dir_z},#{dir_x})")
        slab_solid = _add(f"IFCEXTRUDEDAREASOLID(#{arb_prof},#{slab_ax},#{dir_z},{slab_thick:.6f})")
        slab_shape = _add(f"IFCSHAPEREPRESENTATION(#{ctx},'Body','SweptSolid',(#{slab_solid}))")
        slab_pdef = _add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{slab_shape}))")
        slab_pl = _add(f"IFCLOCALPLACEMENT(#{stry_pl},#{world_axis})")
        slab_id = _add(f"IFCSLAB('0NCKH2SLB0000000000001',$,'FloorSlab',$,$,#{slab_pl},#{slab_pdef},$,.FLOOR.)")
        product_ids.append(slab_id)

    # 3. IfcColumn circular extrusions (from Eq. 4)
    for idx_c, col in enumerate(columns):
        c_pt = _add(f"IFCCARTESIANPOINT(({col.anchor[0]:.6f},{col.anchor[1]:.6f},0.))")
        c_ax = _add(f"IFCAXIS2PLACEMENT3D(#{c_pt},#{dir_z},#{dir_x})")
        c_pl = _add(f"IFCLOCALPLACEMENT(#{stry_pl},#{c_ax})")
        c_prof = _add(f"IFCCIRCLEPROFILEDEF(.AREA.,'ColumnProfile',#{ax_2d},{col.radius:.6f})")
        c_solid = _add(f"IFCEXTRUDEDAREASOLID(#{c_prof},#{world_axis},#{dir_z},{height:.6f})")
        c_shape = _add(f"IFCSHAPEREPRESENTATION(#{ctx},'Body','SweptSolid',(#{c_solid}))")
        c_pdef = _add(f"IFCPRODUCTDEFINITIONSHAPE($,$,(#{c_shape}))")
        col_id = _add(f"IFCCOLUMN('0NCKH2COL{idx_c:013d}',$,'Column_{idx_c}',$,$,#{c_pl},#{c_pdef},$,.COLUMN.)")
        product_ids.append(col_id)

    if product_ids:
        refs = ",".join(f"#{pid}" for pid in product_ids)
        _add(f"IFCRELCONTAINEDINSPATIALSTRUCTURE('0NCKH2CNT0000000000001',$,$,$,({refs}),#{storey})")

    header = [
        "ISO-10303-21;",
        "HEADER;",
        f"FILE_DESCRIPTION(('ViewDefinition [CoordinationView]'),'2;1');",
        f"FILE_NAME('{path.name}','2026-11-25T09:30:00',('nckh_2'),('NCKH_2'),'nckh_2_bim','nckh_2','');",
        f"FILE_SCHEMA(('{schema}'));",
        "ENDSEC;",
        "DATA;",
        *entities,
        "ENDSEC;",
        "END-ISO-10303-21;",
    ]
    path.write_text("\n".join(header) + "\n", encoding="utf-8")
    return path
