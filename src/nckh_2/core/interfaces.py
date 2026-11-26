"""Giao dien truu tuong (Protocols/ABCs) giup tach biet gia dinh don vat the khoi tai tao khong gian."""
from __future__ import annotations

from typing import Protocol, runtime_checkable
from nckh_2.core.types import SceneContext, StageResult


@runtime_checkable
class PipelineStage(Protocol):
    """Giao dien chun cho mot buoc xu ly trong pipeline."""

    stage_id: int
    title: str

    def execute(self, ctx: SceneContext) -> StageResult:
        """Thuc thi buoc xu ly va cap nhat artifacts vao SceneContext."""
        ...


@runtime_checkable
class Masker(Protocol):
    """Giao dien tach nen / phan doan doi tuong."""

    def generate_masks(self, ctx: SceneContext) -> StageResult:
        ...


@runtime_checkable
class Reconstructor(Protocol):
    """Giao dien tai tao SfM / MVS / 3DGS."""

    def reconstruct(self, ctx: SceneContext) -> StageResult:
        ...


@runtime_checkable
class MeshExtractor(Protocol):
    """Giao dien xuat mesh tu point cloud / gaussian splats."""

    def extract_mesh(self, ctx: SceneContext) -> StageResult:
        ...


@runtime_checkable
class MeshRefiner(Protocol):
    """Giao dien tinh chinh & lam min mesh be mat."""

    def refine_mesh(self, ctx: SceneContext) -> StageResult:
        ...
