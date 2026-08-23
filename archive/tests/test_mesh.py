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
    mesh = make_mesh(mode="gossip_uniform")
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


# --- nig_product_consensus: closed-form fusion + live consensus (2026-08) --

def test_nig_product_consensus_matches_weighted_when_consensus_unity():
    """With self.consensus pinned at 1.0 (rho=0, so update_consensus() never
    moves it from its unity init), nig_product_consensus's weights list is
    identical to nig_product_weighted's at every call, so fuse_nig_product()
    must return identical fused parameters -- the closed-form fusion itself is
    unaffected by which mode drives it, only self.consensus's mutability
    differs."""
    env = GridEnvironment(GRID_SIZE, np.full((2, GRID_SIZE, GRID_SIZE), 0.5))
    mesh_w = Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
                  pretrained_path=BASELINE, lr=3e-4, fusion_grid=circular_grid(360),
                  mode="nig_product_weighted", device=torch.device("cpu"), sample_seed=42)
    mesh_c = Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
                  pretrained_path=BASELINE, lr=3e-4, fusion_grid=circular_grid(360),
                  mode="nig_product_consensus", device=torch.device("cpu"),
                  sample_seed=42, rho=0.0)
    contributions = [(1, (0.3, 4.0, 2.5, 1.0)), (2, (-0.6, 2.0, 3.0, 0.8))]
    label_w, cert_w, inh_w = mesh_w._aggregate(contributions)
    label_c, agree_c, inh_c = mesh_c._aggregate(contributions)
    assert label_w == pytest.approx(label_c)
    # self.consensus never updated by a bare _aggregate() call in either mode
    assert all(c == 1.0 for c in mesh_w.consensus.values())
    assert all(c == 1.0 for c in mesh_c.consensus.values())


def test_nig_product_consensus_agreement_drops_under_disagreement():
    env = GridEnvironment(GRID_SIZE, np.full((2, GRID_SIZE, GRID_SIZE), 0.5))
    mesh = Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
                pretrained_path=BASELINE, lr=3e-4, fusion_grid=circular_grid(360),
                mode="nig_product_consensus", device=torch.device("cpu"))
    agreeing = [(1, (0.2, 4.0, 2.5, 1.0)), (2, (0.2, 4.0, 2.5, 1.0))]
    disagreeing = [(1, (0.2, 4.0, 2.5, 1.0)), (2, (-0.9, 4.0, 2.5, 1.0))]
    _, agree_same, _ = mesh._aggregate(agreeing)
    _, agree_diff, _ = mesh._aggregate(disagreeing)
    assert agree_same == pytest.approx(1.0, abs=1e-3)
    assert agree_diff < agree_same


def test_nig_product_consensus_single_contributor_uses_own_consensus_as_inherited():
    env = GridEnvironment(GRID_SIZE, np.full((2, GRID_SIZE, GRID_SIZE), 0.5))
    mesh = Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
                pretrained_path=BASELINE, lr=3e-4, fusion_grid=circular_grid(360),
                mode="nig_product_consensus", device=torch.device("cpu"))
    mesh.consensus[1] = 0.42
    label, agreement, inherited = mesh._aggregate([(1, (0.2, 4.0, 2.5, 1.0))])
    assert label == pytest.approx(0.2)
    assert agreement == 1.0
    assert inherited == pytest.approx(0.42)


def test_nig_product_consensus_mode_tracks_per_node_trust():
    mesh = make_mesh(mode="nig_product_consensus")
    assert all(c == 1.0 for c in mesh.consensus.values())
    for step in range(2):
        mesh.run_timestep([(2.0, 2.0)], step=step)
    assert all(0.0 <= c <= 1.0 for c in mesh.consensus.values())
    assert mesh.consensus[0] > 0.999
    assert any(mesh.consensus[i] < mesh.consensus[0] for i in mesh.nodes if i != 0)


# --- uncertainty_measure ablation (2026-08) ------------------------------

def test_uncertainty_measure_epistemic_is_default_and_matches_unspecified():
    """The ablation's default must reproduce the pre-ablation nig_product
    behaviour exactly -- a Mesh built without specifying uncertainty_measure
    at all must agree bit-for-bit with one that explicitly passes
    "epistemic", since that's the value every prior nig_product/
    nig_product_weighted/nig_product_consensus run implicitly used."""
    env = GridEnvironment(GRID_SIZE, np.full((2, GRID_SIZE, GRID_SIZE), 0.5))
    mesh_default = Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
                        pretrained_path=BASELINE, lr=3e-4, fusion_grid=circular_grid(360),
                        mode="nig_product", device=torch.device("cpu"))
    mesh_explicit = Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
                         pretrained_path=BASELINE, lr=3e-4, fusion_grid=circular_grid(360),
                         mode="nig_product", device=torch.device("cpu"),
                         uncertainty_measure="epistemic")
    assert mesh_default.uncertainty_measure == "epistemic"
    contributions = [(1, (0.2, 4.0, 2.5, 1.0)), (2, (-0.3, 6.0, 3.0, 0.8))]
    assert mesh_default._aggregate(contributions) == mesh_explicit._aggregate(contributions)


def test_uncertainty_measure_changes_fused_certainty_under_disagreement():
    """aleatoric drops the nu (evidence-count) term entirely, so it must
    produce a DIFFERENT fused certainty than epistemic/total once nu varies
    across contributors -- otherwise the ablation parameter would be a no-op."""
    env = GridEnvironment(GRID_SIZE, np.full((2, GRID_SIZE, GRID_SIZE), 0.5))
    contributions = [(1, (0.3, 2.0, 2.5, 1.0)), (2, (-0.4, 9.0, 3.0, 0.4))]
    certs = {}
    for measure in ("epistemic", "aleatoric", "total"):
        mesh = Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
                   pretrained_path=BASELINE, lr=3e-4, fusion_grid=circular_grid(360),
                   mode="nig_product", device=torch.device("cpu"),
                   uncertainty_measure=measure)
        _, cert, _ = mesh._aggregate(contributions)
        certs[measure] = cert
    assert len(set(certs.values())) == 3, certs


def test_uncertainty_measure_rejects_unknown_value():
    env = GridEnvironment(GRID_SIZE, np.full((2, GRID_SIZE, GRID_SIZE), 0.5))
    with pytest.raises(AssertionError):
        Mesh(CENTRES, fov_size=5, grid_size=GRID_SIZE, environment=env,
            pretrained_path=BASELINE, lr=3e-4, fusion_grid=circular_grid(360),
            mode="nig_product", device=torch.device("cpu"), uncertainty_measure="bogus")


# --- node-failure resilience mechanism (2026-08) ------------------------

def test_fail_nodes_removes_from_overlap_graph_symmetrically():
    mesh = make_mesh()
    assert mesh.overlap_graph[1]  # node 1 has neighbours before failing
    neighbours_of_1 = list(mesh.overlap_graph[1])
    mesh.fail_nodes([1])
    assert mesh.overlap_graph[1] == []
    for nb in neighbours_of_1:
        assert 1 not in mesh.overlap_graph[nb]
    assert 1 in mesh.failed_nodes


def test_fail_nodes_is_idempotent():
    mesh = make_mesh()
    mesh.fail_nodes([1])
    mesh.fail_nodes([1])  # must not raise or double-remove
    assert mesh.failed_nodes == {1}


def test_failed_node_never_becomes_anchor_even_if_wearable_in_its_fov():
    mesh = make_mesh()
    mesh.fail_nodes([0])  # node 0 sits exactly at (2, 2)
    covered, _ = mesh.in_coverage([(2.0, 2.0)])
    assert 0 not in covered
    trained = mesh.run_timestep([(2.0, 2.0)], step=0)
    assert 0 not in trained
    assert mesh.nodes[0].cell_beliefs == {}


def test_failed_node_stays_failed_across_timesteps():
    mesh = make_mesh()
    mesh.fail_nodes([0])
    for step in range(2):
        trained = mesh.run_timestep([(2.0, 2.0)], step=step)
        assert 0 not in trained
        assert mesh.nodes[0].cell_beliefs == {}


def test_ever_trained_and_unreachable_nodes_after_failure():
    mesh = make_mesh()
    mesh.run_timestep([(2.0, 2.0)], step=0)
    assert mesh.ever_trained == set(mesh.nodes)  # dense overlap: everyone reached
    assert mesh.unreachable_nodes() == set()
    # fail every neighbour of node 3 so it can never again be anchored or
    # BFS-reached; unreachable_nodes() is defined purely as "surviving but
    # never in ever_trained", so it correctly flags node 3 even before a
    # timestep runs -- there is nothing left that could ever train it.
    mesh2 = make_mesh()
    others = [i for i in mesh2.nodes if i != 3]
    mesh2.fail_nodes(others)
    assert mesh2.unreachable_nodes() == {3}
    mesh2.run_timestep([(20.0, 20.0)], step=0)  # wearable nowhere near node 3's FOV
    assert 3 in mesh2.unreachable_nodes()


def test_connected_components_reflects_failures():
    mesh = make_mesh()
    assert len(mesh.connected_components()) == 1  # dense overlap: one component
    mesh.fail_nodes([1, 2, 3])
    components = mesh.connected_components()
    assert components == [{0}]
    stats = mesh.overlap_graph_stats()
    assert stats["n_surviving"] == 1
    assert stats["n_connected_components"] == 1
    assert stats["mean_degree"] == 0.0
