"""Evaluation package: Tao hinh thong so ky thuat va do luong chat luong tai tao."""
from nckh_2.evaluation.metrics import (
    DetailRetentionMetrics,
    SurfaceMetrics,
    build_synthetic_detailed_box,
    compute_surface_metrics,
    evaluate_detail_preservation,
    evaluate_reconstruction_split,
)
from nckh_2.evaluation.spec_figures import export_all_specification_figures

__all__ = [
    "DetailRetentionMetrics",
    "SurfaceMetrics",
    "build_synthetic_detailed_box",
    "compute_surface_metrics",
    "evaluate_detail_preservation",
    "evaluate_reconstruction_split",
    "export_all_specification_figures",
]
