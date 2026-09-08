"""Width-variant presets for objective #1 (heterogeneous device collaboration,
2026-08). Width-only, per Christian's explicit instruction (depth deferred).

baseline is the untouched architecture (32, 64, 128) that the pretrained
Stage 0 checkpoint was trained under -- only nodes assigned "baseline" can
load that checkpoint; narrow/wide nodes train from scratch (see
MeshNode.load_checkpoint's shape-mismatch handling in mesh.py). narrow/wide
are set at half/double the channel counts of baseline at every conv block.
"""

from __future__ import annotations

WIDTH_VARIANTS: dict[str, tuple[int, int, int]] = {
    "narrow": (16, 32, 64),
    "baseline": (32, 64, 128),
    "wide": (64, 128, 256),
}
