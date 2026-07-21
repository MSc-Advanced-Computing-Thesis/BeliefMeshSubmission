"""Smoke tests for beliefmesh.node.mesh: overlap graph construction, BFS
propagation, full-belief storage (the Defect 1 fix), and frozen-mode inertness.

Small 8x8 grid, 4 nodes, 2 timesteps -- fast enough for the unit suite while
exercising every propagation code path.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.grid import circular_grid
from beliefmesh.node.mesh import Mesh, fov_cells

BASELINE = "runs/stage0/baseline/checkpoints/pretrained_digit7.pth"
pytestmark = pytest.mark.skipif(
    not __import__("pathlib").Path(BASELINE).exists(),
    reason="stage 0 baseline checkpoint not present")

GRID_SIZE = 8
CENTRES = np.array([[2, 2], [5, 2], [2, 5], [5, 5]])


def make_mesh(mode="fusion"):
    env = GridEnvironment(GRID_SIZE, np.full((2, GRID_SIZE, GRID_SIZE), 0.5))
    return Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
                pretrained_path=BASELINE, lr=3e-4,
                fusion_grid=circular_grid(360), mode=mode,
                device=torch.device("cpu"))


def test_fov_cells_clipped_at_boundary():
    cells = fov_cells(0, 0, 5, GRID_SIZE)
    assert all(0 <= r < GRID_SIZE and 0 <= c < GRID_SIZE for r, c in cells)
    assert len(cells) == 9  # 3x3 corner clip of a 5x5 FOV


def test_overlap_graph_symmetric_and_shared_cells_correct():
    mesh = make_mesh()
    for i, neighbours in mesh.overlap_graph.items():
        for j in neighbours:
            assert i in mesh.overlap_graph[j]
            shared = set(mesh.shared_cells[(i, j)])
            assert shared == (mesh.nodes[i].fov_set & mesh.nodes[j].fov_set)
            assert shared  # connected pairs share at least one cell


def test_bfs_propagates_to_all_nodes_with_increasing_hops():
    mesh = make_mesh()
    trained = mesh.run_timestep([(2.0, 2.0)], step=0)
    assert trained == set(mesh.nodes)  # dense overlap: everyone reached
    hops = {i: mesh.nodes[i].hop_distance for i in mesh.nodes}
    assert hops[0] == 0  # node at (2,2) is the anchor
    assert all(h is not None for h in hops.values())
    assert all(h <= 2 for h in hops.values())
    assert any(h >= 1 for h in hops.values())  # someone actually propagated


def test_full_beliefs_stored_uncollapsed():
    mesh = make_mesh()
    mesh.run_timestep([(2.0, 2.0)], step=0)
    for node in mesh.nodes.values():
        assert node.cell_beliefs, "every trained node must hold beliefs"
        for cell, belief in node.cell_beliefs.items():
            assert len(belief) == 4  # (gamma, nu, alpha, beta) -- never collapsed
            g, n, a, b = belief
            assert n > 0 and a > 1 and b > 0


def test_frozen_mode_never_updates_weights():
    mesh = make_mesh(mode="frozen")
    before = {i: [p.clone() for p in node.model.parameters()]
              for i, node in mesh.nodes.items()}
    mesh.run_timestep([(2.0, 2.0)], step=0)
    for i, node in mesh.nodes.items():
        for p0, p1 in zip(before[i], node.model.parameters()):
            assert torch.equal(p0, p1)


def test_naive_and_fusion_modes_both_run_and_evaluate():
    for mode in ("naive", "fusion"):
        mesh = make_mesh(mode=mode)
        mesh.run_timestep([(2.0, 2.0)], step=0)
        metrics = mesh.evaluate(step=0)
        assert len(metrics) == len(mesh.nodes)
        for m in metrics.values():
            assert m["mse"] is not None and np.isfinite(m["mse"])
            assert m["certainty"] is not None and 0 < m["certainty"] <= 1


def test_fedavg_mode_replaces_weights_with_neighbour_mean():
    mesh = make_mesh(mode="fedavg")
    before = {i: [p.clone() for p in node.model.parameters()]
              for i, node in mesh.nodes.items()}
    mesh.run_timestep([(2.0, 2.0)], step=0)
    # anchor (node 0) trained on ground truth; at least one non-anchor must have
    # had its weights REPLACED (parameter exchange), not gradient-updated
    changed = [i for i, node in mesh.nodes.items()
               if any(not torch.equal(b, p) for b, p in zip(before[i], node.model.parameters()))]
    assert 0 in changed
    assert len(changed) > 1


def test_consensus_mode_tracks_per_node_trust():
    mesh = make_mesh(mode="consensus")
    assert all(c == 1.0 for c in mesh.consensus.values())  # unity init
    for step in range(2):
        mesh.run_timestep([(2.0, 2.0)], step=step)
    assert all(0.0 <= c <= 1.0 for c in mesh.consensus.values())
    # anchor (node 0 at the wearable) stays at unity; downstream nodes, whose
    # early contributors disagree at least somewhat, drift below it
    assert mesh.consensus[0] > 0.999
    assert any(mesh.consensus[i] < mesh.consensus[0] for i in mesh.nodes if i != 0)
