"""Spatial mesh: overlap graph + breadth-first belief propagation. Spec Sec 7-8.

The Defect 1 fix lives here: nodes store and exchange FULL (gamma, nu, alpha,
beta) beliefs per cell. Fusion consumes them via product_of_experts.fuse();
nothing is collapsed to a (pred, uncertainty, certainty) point estimate on the
propagation path. Certainty summaries exist only for evaluation maps.

Aggregation modes:
  fusion -- corrected N-way product-of-experts on the circular grid (unweighted;
            consensus tempering is a later, separate condition -- Spec Sec 9 step 5)
  naive  -- plain mean of contributor gammas (the comparison arm)
  frozen -- no training ever; pretrained predictions only (baseline arm)

Model: plain EvidentialCNN, all weights trainable. Deliberate deviation from
the prior repo's SpatialEvidentialCNN (coordinate-conditioned): the spec's
Sec 3 model description carries no coordinate input, and the deviation is
recorded rather than concealed.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np
import torch

from beliefmesh.data.grid_environment import GridEnvironment
from beliefmesh.fusion.product_of_experts import fuse
from beliefmesh.metrics.circular import circular_diff
from beliefmesh.models.evidential import nig_loss, predictive_uncertainty
from beliefmesh.models.evidential_cnn import EvidentialCNN


def fov_cells(cx: int, cy: int, fov_size: int, grid_size: int) -> list[tuple[int, int]]:
    half = fov_size // 2
    return [(row, col)
            for row in range(max(0, cy - half), min(grid_size, cy + half + 1))
            for col in range(max(0, cx - half), min(grid_size, cx + half + 1))]


class MeshNode:
    """One node: position, field of view, model, and per-timestep belief state."""

    def __init__(self, node_id: int, centre: tuple[int, int],
                 cells: list[tuple[int, int]], lr: float, device: torch.device):
        self.node_id = node_id
        self.centre = centre
        self.fov_cells = cells
        self.fov_set = set(cells)
        self.device = device
        self.model = EvidentialCNN().to(device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.hop_distance: int | None = None
        # cell -> (gamma, nu, alpha, beta) floats: the FULL belief, uncollapsed
        self.cell_beliefs: dict[tuple[int, int], tuple[float, float, float, float]] = {}

    def load_checkpoint(self, path: str | Path):
        self.model.load_state_dict(torch.load(path, map_location=self.device))

    def reset_timestep(self):
        self.hop_distance = None
        self.cell_beliefs = {}

    def train_on(self, images: list[torch.Tensor], targets: torch.Tensor) -> float | None:
        if not images:
            return None
        batch = torch.stack(images).to(self.device)
        targets = targets.to(self.device)
        self.optimiser.zero_grad()
        gamma, nu, alpha, beta = self.model(batch)
        loss = nig_loss(gamma, nu, alpha, beta, targets)
        loss.backward()
        self.optimiser.step()
        return loss.item()

    def predict_cells(self, images: list[torch.Tensor], keys: list[tuple[int, int]]):
        """Predict and store FULL beliefs for the given cells."""
        if not images:
            return
        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(batch)
        for i, key in enumerate(keys):
            self.cell_beliefs[key] = (gamma[i].item(), nu[i].item(),
                                      alpha[i].item(), beta[i].item())


class Mesh:
    """Overlap-graph mesh with breadth-first belief propagation per timestep."""

    def __init__(self, node_centres: np.ndarray, fov_size: int, grid_size: int,
                 environment: GridEnvironment, pretrained_path: str | Path,
                 lr: float, fusion_grid: torch.Tensor, mode: str = "fusion",
                 device: torch.device | None = None, sample_seed: int = 42,
                 rho: float = 0.2):
        assert mode in ("fusion", "naive", "frozen", "consensus")
        self.mode = mode
        self.environment = environment
        self.grid_size = grid_size
        self.fusion_grid = fusion_grid
        self.sample_rng = np.random.default_rng(sample_seed)
        # per-node running consensus (Spec Sec 6.2); only consulted in
        # consensus mode. Initialised at unity: every node starts from the
        # same trusted pretrained checkpoint.
        self.rho = rho
        self.consensus: dict[int, float] = {}
        device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device

        self.nodes: dict[int, MeshNode] = {}
        for i, (cx, cy) in enumerate(node_centres):
            node = MeshNode(i, (int(cx), int(cy)),
                            fov_cells(int(cx), int(cy), fov_size, grid_size),
                            lr=lr, device=device)
            node.load_checkpoint(pretrained_path)
            if mode == "frozen":
                node.model.eval()
            self.nodes[i] = node
            self.consensus[i] = 1.0

        # overlap graph + shared cells between connected pairs
        self.overlap_graph: dict[int, list[int]] = {i: [] for i in self.nodes}
        self.shared_cells: dict[tuple[int, int], list[tuple[int, int]]] = {}
        ids = list(self.nodes)
        for a in range(len(ids)):
            for b in range(a + 1, len(ids)):
                i, j = ids[a], ids[b]
                shared = list(self.nodes[i].fov_set & self.nodes[j].fov_set)
                if shared:
                    self.overlap_graph[i].append(j)
                    self.overlap_graph[j].append(i)
                    self.shared_cells[(i, j)] = shared
                    self.shared_cells[(j, i)] = shared

    def in_coverage(self, wearable_positions) -> tuple[set[int], list[tuple[int, int]]]:
        cells = [(int(p[0]), int(p[1])) for p in wearable_positions]
        covered = {i for i, node in self.nodes.items()
                   if any(c in node.fov_set for c in cells)}
        return covered, cells

    def _aggregate(self, contributions: list[tuple[int, tuple[float, float, float, float]]]
                   ) -> tuple[float, float, float]:
        """Aggregate contributors' FULL beliefs for one cell into a scalar
        pseudo-label (Spec Sec 5: the fused mode propagates, nothing else).

        contributions: list of (contributor_id, (gamma, nu, alpha, beta)).
        Returns (pseudo_label, agreement, inherited_trust); the latter two are
        meaningful only in consensus mode (1.0 placeholders otherwise).
        """
        from beliefmesh.fusion.consensus import agreement_score, inherited_trust

        beliefs_raw = [b for _, b in contributions]
        if self.mode == "naive":
            return float(np.mean([b[0] for b in beliefs_raw])), 1.0, 1.0

        c_vals = torch.tensor([self.consensus[cid] for cid, _ in contributions])
        if len(contributions) == 1:
            # single contributor: its mode is the label, no disagreement exists
            inherited = float(c_vals[0]) if self.mode == "consensus" else 1.0
            return beliefs_raw[0][0], 1.0, inherited

        beliefs = [tuple(torch.tensor([v]) for v in b) for b in beliefs_raw]
        if self.mode == "consensus":
            weights = c_vals.clamp(min=1e-3)  # all-zero weights would degenerate argmax
            modes, logp = fuse(beliefs, self.fusion_grid.cpu(), weights=weights)
            return modes.item(), agreement_score(logp).item(), inherited_trust(c_vals)
        modes, _ = fuse(beliefs, self.fusion_grid.cpu())
        return modes.item(), 1.0, 1.0

    def run_timestep(self, wearable_positions, step: int,
                     n_wearable_samples: int = 1, n_train_repeats: int = 1) -> set[int]:
        for node in self.nodes.values():
            node.reset_timestep()

        anchors, wearable_cells = self.in_coverage(wearable_positions)
        trained: set[int] = set()
        queue: deque[int] = deque()

        # hop 0: anchors train on wearable-cell ground truth (unless frozen)
        for node_id in anchors:
            node = self.nodes[node_id]
            anchor_cells = [c for c in wearable_cells if c in node.fov_set]
            if not anchor_cells:
                continue
            if self.mode != "frozen":
                images, targets = [], []
                for cell in anchor_cells:
                    imgs, tgts = self.environment.get_multiple_rotations(
                        cell, step, n=n_wearable_samples, rng=self.sample_rng)
                    images.extend(imgs)
                    targets.extend(tgts)
                for _ in range(n_train_repeats):
                    node.train_on(images, torch.stack(targets))
                if self.mode == "consensus":
                    # anchor timestep: no fusion, no disagreement -- consensus
                    # updates toward unity (Spec Sec 6.3)
                    from beliefmesh.fusion.consensus import update_consensus
                    self.consensus[node_id] = update_consensus(
                        self.consensus[node_id], 1.0, self.rho)
            imgs, _, keys = self.environment.get_batch_for_cells(node.fov_cells, step)
            node.predict_cells(imgs, keys)
            node.hop_distance = 0
            trained.add(node_id)
            queue.append(node_id)

        # BFS outward: each untrained neighbour collects FULL beliefs on shared
        # cells from ALL already-trained neighbours, aggregates, trains, predicts
        while queue:
            current_id = queue.popleft()
            current_hop = self.nodes[current_id].hop_distance or 0

            for neighbour_id in self.overlap_graph[current_id]:
                if neighbour_id in trained:
                    continue
                neighbour = self.nodes[neighbour_id]

                supervision: dict[tuple[int, int], list] = {}
                for other_id in self.overlap_graph[neighbour_id]:
                    if other_id not in trained:
                        continue
                    other = self.nodes[other_id]
                    for cell in self.shared_cells.get((other_id, neighbour_id), []):
                        if cell in other.cell_beliefs:
                            supervision.setdefault(cell, []).append(
                                (other_id, other.cell_beliefs[cell]))
                if not supervision:
                    continue

                if self.mode != "frozen":
                    images, labels, cell_trusts = [], [], []
                    for cell, contributions in supervision.items():
                        img, _ = self.environment.get_cell_input(cell[0], cell[1], step)
                        images.append(img)
                        label, agreement, inherited = self._aggregate(contributions)
                        labels.append(label)
                        cell_trusts.append(agreement * inherited)
                    for _ in range(n_train_repeats):
                        neighbour.train_on(images, torch.tensor(labels, dtype=torch.float32))
                    if self.mode == "consensus":
                        # c_new = mean over fused cells of agreement * inherited
                        # (Spec Sec 6.2), EMA'd into the receiver's consensus
                        from beliefmesh.fusion.consensus import update_consensus
                        self.consensus[neighbour_id] = update_consensus(
                            self.consensus[neighbour_id],
                            float(np.mean(cell_trusts)), self.rho)

                imgs, _, keys = self.environment.get_batch_for_cells(neighbour.fov_cells, step)
                neighbour.predict_cells(imgs, keys)

                hop = current_hop + 1
                if neighbour.hop_distance is None or hop < neighbour.hop_distance:
                    neighbour.hop_distance = hop
                trained.add(neighbour_id)
                queue.append(neighbour_id)

        return trained

    def evaluate(self, step: int) -> dict:
        """Per-node wrapped MSE and mean display-certainty over stored beliefs."""
        out = {}
        for node_id, node in self.nodes.items():
            if not node.cell_beliefs:
                out[node_id] = {"mse": None, "certainty": None,
                                "hop_distance": node.hop_distance}
                continue
            errs, certs = [], []
            for cell, (g, n, a, b) in node.cell_beliefs.items():
                _, truth = self.environment.get_cell_input(cell[0], cell[1], step)
                diff = circular_diff(torch.tensor(g), truth)
                errs.append(diff.item() ** 2)
                unc = predictive_uncertainty(
                    torch.tensor(n), torch.tensor(a), torch.tensor(b)).clamp(max=10.0)
                certs.append(1.0 / (1.0 + unc.item()))
            out[node_id] = {"mse": float(np.mean(errs)),
                            "certainty": float(np.mean(certs)),
                            "hop_distance": node.hop_distance}
        return out
