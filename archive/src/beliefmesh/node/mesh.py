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
  fedavg_global   -- canonical FedAvg: one shared model, averaged from this
                      round's anchors and broadcast to every node.
  gossip_uniform  -- the decentralised PARAMETER-exchange comparator (Christian's
                      objective #2, 2026-08): a receiver replaces its weights
                      with the unweighted mean of its already-trained overlap
                      neighbours' full model state_dicts. No beliefs cross the
                      link. Formerly named "fedavg" -- renamed to avoid
                      confusion with fedavg_global, which is a different
                      mechanism (single global model, not per-edge gossip).
  gossip_weighted -- same as gossip_uniform, but the neighbour average is
                      weighted by each contributor's MeshNode.n_samples_trained
                      (FedAvg-style weighting by local data quantity, McMahan
                      et al. 2017) instead of uniform -- the "stronger"
                      parameter-exchange baseline.

Communication-volume instrumentation (objective #2): self.comm_bytes_step /
self.comm_bytes_cumulative track bytes notionally transmitted this step / over
the whole run, assuming float32. Belief-exchange modes count 16 bytes (4 NIG
floats) per (contributor, shared cell) pair actually consumed; gossip/fedavg
modes count each contributor's full model size in bytes per transmission.
Per-node model size (not a single constant) so this stays correct once nodes
have heterogeneous architectures (objective #1).

Model: EvidentialCNN + 2 CoordConv-style input channels encoding each cell's
position WITHIN the node's own FOV (local, not global -- Christian's call
after the multi-region diagnostic empirically confirmed a node whose FOV
spans two genuinely different true offset regions cannot represent both:
two images from different cells within one FOV were otherwise
indistinguishable to the model, so it collapses to a single compromise
biased toward whichever region dominates its training exposure and is
confidently wrong on the other. This is DIFFERENT from the prior repo's
SpatialEvidentialCNN, which conditioned on GLOBAL grid position -- that
was deliberately excluded per spec Sec 3 and stays excluded; a node still
has no idea where it sits in the wider grid, only where a given cell sits
relative to its own FOV, which is what a real deployed drone would
plausibly know without any global coordination.
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


def local_coord_channels(cell: tuple[int, int], centre: tuple[int, int],
                         fov_size: int, image_hw: tuple[int, int]) -> torch.Tensor:
    """2 constant-value channels of shape (2, H, W): the cell's (dy, dx)
    position relative to the node's OWN FOV centre, normalised to [-1, 1]
    (CoordConv, Liu et al. 2018). Local to this node only -- no global grid
    position is ever exposed to the model."""
    half = fov_size // 2
    row, col = cell
    cx, cy = centre
    dy = (row - cy) / max(half, 1)
    dx = (col - cx) / max(half, 1)
    h, w = image_hw
    return torch.stack([
        torch.full((h, w), float(dy)),
        torch.full((h, w), float(dx)),
    ])


class MeshNode:
    """One node: position, field of view, model, and per-timestep belief state."""

    def __init__(self, node_id: int, centre: tuple[int, int],
                 cells: list[tuple[int, int]], lr: float, device: torch.device,
                 fov_size: int = 7):
        self.node_id = node_id
        self.centre = centre
        self.fov_cells = cells
        self.fov_set = set(cells)
        self.fov_size = fov_size
        self.device = device
        self.model = EvidentialCNN(in_channels=5).to(device)
        self.optimiser = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.hop_distance: int | None = None
        # cell -> (gamma, nu, alpha, beta) floats: the FULL belief, uncollapsed
        self.cell_beliefs: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        # persists ACROSS timesteps (cell_beliefs above is wiped every step) --
        # a node's own most recent belief per cell, for self-consistency
        # fusion (Spec: fusion mode only, Christian's echo-chamber-drift fix).
        # Without this a receiving node has no memory of its own prior stance
        # and gets fully overwritten toward whatever a single neighbour
        # currently believes -- a random walk with no restoring force.
        self.last_beliefs: dict[tuple[int, int], tuple[float, float, float, float]] = {}
        # cumulative count of (image, target) pairs this node has ever been
        # trained on, across the whole run -- used ONLY by
        # mode="gossip_weighted" to reproduce FedAvg's sample-count weighting
        # (Sec 2, McMahan et al. 2017) at the per-edge/per-neighbour level.
        # Incremented once per train_on() call by the batch size actually
        # trained on that call, so a cell visited under n_train_repeats>1
        # counts each repeat -- it measures accumulated training WORK, not
        # distinct data, a deliberate reading of "accumulated training
        # sample count" (a node that has been retrained on the same anchor
        # repeatedly has genuinely put more gradient steps into its weights
        # than one that hasn't, which is what the receiving neighbour should
        # trust more under this weighting scheme).
        self.n_samples_trained: int = 0

    def load_checkpoint(self, path: str | Path):
        state_dict = torch.load(path, map_location=self.device)
        conv1_w = state_dict.get("conv1.weight")
        if conv1_w is not None and conv1_w.shape[1] != self.model.conv1.in_channels:
            # pretrained checkpoint predates the CoordConv channels (3 in,
            # not 5) -- expand conv1's weight with ZERO-initialised slices
            # for the 2 new coord channels, so the model starts out
            # numerically IDENTICAL to the old 3-channel behaviour
            # regardless of what coordinate values are fed in, and only
            # learns to use the coord signal as training proceeds.
            extra = self.model.conv1.in_channels - conv1_w.shape[1]
            assert extra > 0, f"unexpected conv1 channel shrink: {conv1_w.shape[1]} -> {self.model.conv1.in_channels}"
            pad = torch.zeros(conv1_w.shape[0], extra, *conv1_w.shape[2:],
                              device=conv1_w.device, dtype=conv1_w.dtype)
            state_dict = dict(state_dict)
            state_dict["conv1.weight"] = torch.cat([conv1_w, pad], dim=1)
        self.model.load_state_dict(state_dict)

    def reset_timestep(self):
        self.hop_distance = None
        self.cell_beliefs = {}

    def _with_coords(self, images: list[torch.Tensor],
                     keys: list[tuple[int, int]]) -> list[torch.Tensor]:
        h, w = images[0].shape[-2:]
        return [torch.cat([img, local_coord_channels(key, self.centre, self.fov_size, (h, w))])
                for img, key in zip(images, keys)]

    def train_on(self, images: list[torch.Tensor], targets: torch.Tensor,
                lam: float = 0.1, weights: torch.Tensor | None = None,
                keys: list[tuple[int, int]] | None = None) -> float | None:
        if not images:
            return None
        if keys is not None:
            images = self._with_coords(images, keys)
        batch = torch.stack(images).to(self.device)
        targets = targets.to(self.device)
        if weights is not None:
            weights = weights.to(self.device)
        self.optimiser.zero_grad()
        gamma, nu, alpha, beta = self.model(batch)
        loss = nig_loss(gamma, nu, alpha, beta, targets, lam=lam, weights=weights)
        loss.backward()
        self.optimiser.step()
        self.n_samples_trained += len(images)
        return loss.item()

    def predict_cells(self, images: list[torch.Tensor], keys: list[tuple[int, int]]):
        """Predict and store FULL beliefs for the given cells."""
        if not images:
            return
        images = self._with_coords(images, keys)
        batch = torch.stack(images).to(self.device)
        with torch.no_grad():
            gamma, nu, alpha, beta = self.model(batch)
        for i, key in enumerate(keys):
            belief = (gamma[i].item(), nu[i].item(), alpha[i].item(), beta[i].item())
            self.cell_beliefs[key] = belief
            self.last_beliefs[key] = belief


class Mesh:
    """Overlap-graph mesh with breadth-first belief propagation per timestep."""

    def __init__(self, node_centres: np.ndarray, fov_size: int, grid_size: int,
                 environment: GridEnvironment, pretrained_path: str | Path,
                 lr: float, fusion_grid: torch.Tensor, mode: str = "fusion",
                 device: torch.device | None = None, sample_seed: int = 42,
                 rho: float = 0.2, lam: float = 0.1, temper_gradient: bool = True,
                 self_weight: float = 0.0):
        assert mode in ("fusion", "naive", "frozen", "consensus", "gossip_uniform",
                        "gossip_weighted", "fedavg_global", "certainty")
        self.mode = mode
        self.lam = lam
        # weight given to a node's OWN last belief for a cell (self.last_beliefs)
        # as an extra fused contributor alongside external neighbours, redoing
        # the earlier self-consistency idea as a TUNABLE, capped weight instead
        # of full-weight injection (which tested 5x worse -- see _aggregate's
        # docstring and mesh.py module history). 0.0 (default) reproduces prior
        # behaviour exactly; fusion/consensus modes only.
        #
        # SWEPT AND REJECTED (2026-07-31, static-spatial v2 world, lr=3e-5,
        # lam=0.1): whole_run MSE degrades MONOTONICALLY with self_weight --
        # 0.0289 (0.0) -> 0.0304 (0.05) -> 0.0316 (0.1) -> 0.0385 (0.3) ->
        # 0.0607 (0.5) -> 0.1688 (1.0), where cert-MSE r also flips from
        # -0.57 to +0.07 (confidently wrong). No sweet spot at any weight
        # tested; even small amounts hurt. Confirms the original full-
        # injection failure was the reinforcement-loop mechanism itself, not
        # just an overly aggressive weight -- a node's own recent output is
        # not a safe fusion input at any capped strength found so far. The
        # "correct belief gets overwritten while the wearable is elsewhere"
        # problem this was meant to fix is better attacked via routing
        # (see wearable_policy="zoned_epistemic", runner.py) -- getting the
        # wearable back before the belief decays, rather than making the
        # decayed belief resist being corrected.
        self.self_weight = self_weight
        # consensus-tempered gradient toggle: when False, propagation training
        # reduces to the pre-tempering behaviour (unweighted mean loss), for a
        # clean A/B isolating tempering's own effect from lam/rho.
        self.temper_gradient = temper_gradient
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
        # communication-volume instrumentation (objective #2) -- see module
        # docstring. comm_bytes_step is reset at the top of every
        # run_timestep(); comm_bytes_cumulative accumulates across the run.
        self.comm_bytes_step: int = 0
        self.comm_bytes_cumulative: int = 0

        self.nodes: dict[int, MeshNode] = {}
        for i, (cx, cy) in enumerate(node_centres):
            node = MeshNode(i, (int(cx), int(cy)),
                            fov_cells(int(cx), int(cy), fov_size, grid_size),
                            lr=lr, device=device, fov_size=fov_size)
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

    def _model_bytes(self, node_id: int) -> int:
        """float32 size of node_id's model -- per-node, not a shared constant,
        so this stays correct once nodes have heterogeneous architectures
        (objective #1)."""
        return sum(p.numel() for p in self.nodes[node_id].model.parameters()) * 4

    def in_coverage(self, wearable_positions) -> tuple[set[int], list[tuple[int, int]]]:
        cells = [(int(p[0]), int(p[1])) for p in wearable_positions]
        covered = {i for i, node in self.nodes.items()
                   if any(c in node.fov_set for c in cells)}
        return covered, cells

    def _aggregate(self, contributions: list[tuple[int, tuple[float, float, float, float]]],
                   own_belief: tuple[float, float, float, float] | None = None,
                   ) -> tuple[float, float, float]:
        """Aggregate contributors' FULL beliefs for one cell into a scalar
        pseudo-label (Spec Sec 5: the fused mode propagates, nothing else).

        contributions: list of (contributor_id, (gamma, nu, alpha, beta)).
        own_belief: the receiving node's own last belief for this cell
        (self.last_beliefs), included as an extra weighted contributor at
        weight self.self_weight when > 0 (fusion/consensus only -- see
        Mesh.self_weight docstring). None/0.0 reproduces prior behaviour
        exactly.
        Returns (pseudo_label, agreement, inherited_trust); the latter two are
        meaningful only in consensus mode (1.0 placeholders otherwise).
        """
        from beliefmesh.fusion.consensus import agreement_score, inherited_trust
        from beliefmesh.models.evidential import predictive_uncertainty

        beliefs_raw = [b for _, b in contributions]
        if self.mode == "naive":
            return float(np.mean([b[0] for b in beliefs_raw])), 1.0, 1.0

        if self.mode == "certainty":
            # DELIBERATE ablation reproducing the prior repo's actual
            # mechanism (a scalar certainty-weighted average of point
            # estimates), done correctly this time (certainty properly wired
            # in, not just computed and discarded). NOT true Bayesian fusion
            # -- collapses each belief to (gamma, certainty) before combining,
            # which is exactly Defect 1 from the audit. Exists only as a
            # same-codebase, same-environment control against `consensus`/
            # `fusion`, isolating the aggregation rule as the only variable.
            gammas = np.array([b[0] for b in beliefs_raw])
            uncs = np.array([predictive_uncertainty(
                torch.tensor([b[1]]), torch.tensor([b[2]]), torch.tensor([b[3]])
            ).clamp(max=10.0).item() for b in beliefs_raw])
            certs = 1.0 / (1.0 + uncs)
            return float(np.sum(certs * gammas) / np.sum(certs)), 1.0, 1.0

        include_self = self.self_weight > 0 and own_belief is not None
        c_vals = torch.tensor([self.consensus[cid] for cid, _ in contributions])
        if len(contributions) == 1 and not include_self:
            # single contributor: its mode is the label, no disagreement exists
            inherited = float(c_vals[0]) if self.mode == "consensus" else 1.0
            return beliefs_raw[0][0], 1.0, inherited

        beliefs = [tuple(torch.tensor([v]) for v in b) for b in beliefs_raw]
        if self.mode == "consensus":
            weights = c_vals.clamp(min=1e-3)  # all-zero weights would degenerate argmax
            if include_self:
                beliefs = beliefs + [tuple(torch.tensor([v]) for v in own_belief)]
                weights = torch.cat([weights, torch.tensor([self.self_weight])])
            modes, logp = fuse(beliefs, self.fusion_grid.cpu(), weights=weights)
            return modes.item(), agreement_score(logp).item(), inherited_trust(c_vals)
        if include_self:
            beliefs = beliefs + [tuple(torch.tensor([v]) for v in own_belief)]
            weights = torch.cat([torch.ones(len(beliefs_raw)), torch.tensor([self.self_weight])])
            modes, _ = fuse(beliefs, self.fusion_grid.cpu(), weights=weights)
        else:
            modes, _ = fuse(beliefs, self.fusion_grid.cpu())
        return modes.item(), 1.0, 1.0

    def run_timestep(self, wearable_positions, step: int,
                     n_wearable_samples: int = 1, n_train_repeats: int = 1) -> set[int]:
        for node in self.nodes.values():
            node.reset_timestep()
        self.comm_bytes_step = 0

        anchors, wearable_cells = self.in_coverage(wearable_positions)
        trained: set[int] = set()
        queue: deque[int] = deque()

        if self.mode == "fedavg_global":
            # CANONICAL FedAvg round: anchors (this round's participating
            # clients) train locally from the shared global weights; the new
            # global model is the mean of the participants' models and is
            # distributed to EVERY node. One shared model, no spatial
            # specialisation -- red- and blue-adapted updates merge whenever
            # multiple anchors occupy different condition regions.
            participant_models = []
            for node_id in anchors:
                node = self.nodes[node_id]
                anchor_cells = [c for c in wearable_cells if c in node.fov_set]
                if not anchor_cells:
                    continue
                images, targets, keys = [], [], []
                for cell in anchor_cells:
                    imgs, tgts = self.environment.get_multiple_rotations(
                        cell, step, n=n_wearable_samples, rng=self.sample_rng)
                    images.extend(imgs)
                    targets.extend(tgts)
                    keys.extend([cell] * len(imgs))
                for _ in range(n_train_repeats):
                    node.train_on(images, torch.stack(targets), lam=self.lam, keys=keys)
                participant_models.append(node.model.state_dict())
                node.hop_distance = 0
                trained.add(node_id)
                queue.append(node_id)
            if participant_models:
                with torch.no_grad():
                    global_sd = {k: torch.stack([sd[k] for sd in participant_models]).mean(0)
                                 for k in participant_models[0]}
                    for node in self.nodes.values():
                        node.model.load_state_dict(global_sd)
                # comm: every participant uploads its full model to the
                # aggregator, which broadcasts the merged model to EVERY
                # node (not just participants) -- the "no spatial
                # specialisation" cost is also a communication cost.
                participant_bytes = sum(self._model_bytes(nid) for nid in trained)
                broadcast_bytes = sum(self._model_bytes(nid) for nid in self.nodes)
                self.comm_bytes_step += participant_bytes + broadcast_bytes
            # BFS purely to assign hop labels and produce per-node predictions
            while queue:
                current_id = queue.popleft()
                current_hop = self.nodes[current_id].hop_distance or 0
                for neighbour_id in self.overlap_graph[current_id]:
                    if neighbour_id in trained:
                        continue
                    neighbour = self.nodes[neighbour_id]
                    hop = current_hop + 1
                    if neighbour.hop_distance is None or hop < neighbour.hop_distance:
                        neighbour.hop_distance = hop
                    trained.add(neighbour_id)
                    queue.append(neighbour_id)
            for node_id in trained:
                node = self.nodes[node_id]
                imgs, _, keys = self.environment.get_batch_for_cells(node.fov_cells, step)
                node.predict_cells(imgs, keys)
            self.comm_bytes_cumulative += self.comm_bytes_step
            return trained

        # hop 0: anchors train on wearable-cell ground truth (unless frozen)
        for node_id in anchors:
            node = self.nodes[node_id]
            anchor_cells = [c for c in wearable_cells if c in node.fov_set]
            if not anchor_cells:
                continue
            if self.mode != "frozen":
                images, targets, keys = [], [], []
                for cell in anchor_cells:
                    imgs, tgts = self.environment.get_multiple_rotations(
                        cell, step, n=n_wearable_samples, rng=self.sample_rng)
                    images.extend(imgs)
                    targets.extend(tgts)
                    keys.extend([cell] * len(imgs))
                for _ in range(n_train_repeats):
                    node.train_on(images, torch.stack(targets), lam=self.lam, keys=keys)
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

                if self.mode in ("gossip_uniform", "gossip_weighted"):
                    # PARAMETER exchange comparator (objective #2, 2026-08):
                    # the receiver replaces its weights with an average of its
                    # already-trained overlap neighbours' full model
                    # state_dicts. No beliefs cross this link at all -- the
                    # ONLY thing that differs from the belief-exchange arms
                    # below is the unit of exchange, everything else (node
                    # placement, wearable motion, schedule, BFS order, lr,
                    # optimiser) is identical, per Christian's fairness spec.
                    # gossip_uniform: uniform mean. gossip_weighted: weighted
                    # by each contributor's accumulated n_samples_trained
                    # (FedAvg-style, the "stronger" baseline).
                    #
                    # n_samples_trained counts ONLY direct ground-truth
                    # training (incremented inside train_on(), which a gossip
                    # receiver never calls -- it gets load_state_dict'd, not
                    # trained). Deliberately NOT propagated/summed forward
                    # into the receiver here: an earlier version set
                    # neighbour.n_samples_trained = sum(contributor counts)
                    # after every average, intending it to compound across
                    # hops the way belief certainty does -- but the overlap
                    # graph is cyclic (mean degree ~15, not a tree), so the
                    # same upstream anchor visits get re-summed through
                    # multiple paths every timestep and compound
                    # combinatorially: measured 4.8e11 after 40 steps on a
                    # 36-node grid, a nonsense number. A node's weight in this
                    # scheme is therefore its own direct anchor history only
                    # -- 0 for a node that has never itself been an anchor,
                    # which correctly falls back to uniform (all-zero clause
                    # below) rather than fabricating a multi-hop count that
                    # cannot be computed soundly on a cyclic graph without
                    # double-counting.
                    contributor_ids = sorted({cid for lst in supervision.values()
                                              for cid, _ in lst})
                    with torch.no_grad():
                        state_dicts = [self.nodes[cid].model.state_dict()
                                       for cid in contributor_ids]
                        if self.mode == "gossip_weighted":
                            n = torch.tensor([float(self.nodes[cid].n_samples_trained)
                                              for cid in contributor_ids])
                            w = (n / n.sum()) if n.sum() > 0 else torch.full_like(n, 1 / len(n))
                            averaged = {k: sum(w[i] * sd[k] for i, sd in enumerate(state_dicts))
                                        for k in state_dicts[0]}
                        else:
                            averaged = {k: torch.stack([sd[k] for sd in state_dicts]).mean(0)
                                        for k in state_dicts[0]}
                        neighbour.model.load_state_dict(averaged)
                    # comm: each contributor uploads its full model to the receiver
                    self.comm_bytes_step += sum(self._model_bytes(cid) for cid in contributor_ids)
                elif self.mode != "frozen":
                    images, labels, cell_trusts, prop_keys = [], [], [], []
                    for cell, contributions in supervision.items():
                        img, _ = self.environment.get_cell_input(cell[0], cell[1], step)
                        images.append(img)
                        prop_keys.append(cell)
                        own_belief = (neighbour.last_beliefs.get(cell)
                                      if self.self_weight > 0 else None)
                        label, agreement, inherited = self._aggregate(contributions, own_belief)
                        labels.append(label)
                        cell_trusts.append(agreement * inherited)
                        # comm: each contributor sends its full (gamma, nu,
                        # alpha, beta) belief for this cell -- 4 float32s
                        self.comm_bytes_step += len(contributions) * 4 * 4
                    # consensus-tempered gradient: a low-trust fused label
                    # produces a smaller effective update than a fully-trusted
                    # one, instead of always training at full strength on
                    # whatever fusion produced (Sec 6.2's cell_trusts, until
                    # now only used for the EMA update below and then
                    # discarded). No-op outside consensus mode: fusion/naive's
                    # _aggregate() always returns agreement=inherited=1.0, so
                    # cell_trusts is uniformly 1.0 there -- ordinary unweighted
                    # mean, unchanged behaviour.
                    trust_weights = (torch.tensor(cell_trusts, dtype=torch.float32)
                                      if self.temper_gradient else None)
                    for _ in range(n_train_repeats):
                        neighbour.train_on(images, torch.tensor(labels, dtype=torch.float32),
                                          lam=self.lam, weights=trust_weights, keys=prop_keys)
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

        self.comm_bytes_cumulative += self.comm_bytes_step
        return trained

    def epistemic_map(self) -> np.ndarray:
        """(H, W) mean epistemic uncertainty (1/nu) per cell over the nodes
        currently holding beliefs about it; NaN where no beliefs exist. Used
        by the uncertainty-directed wearable policy -- this is the epistemic
        map the evidential head was chosen to provide (interim Sec 2.4).

        NOTE for interpreting routing videos (2026-07-31): this is NOT the
        same quantity as the "Certainty" panel generate_video.py/runner.py
        plot, which is 1/(1+predictive_uncertainty) with predictive_uncertainty
        = beta/(nu*(alpha-1)) -- a combined evidence-count-AND-difficulty
        measure, not pure evidence count. A cell can look "reddest" (least
        certain) on that panel without being the lowest-nu cell this map
        would route to, and vice versa. The panel is ALSO a rolling average
        over the last N steps (--rolling, default 50), while a routing
        decision under wearable_policy="uncertainty_guided" is made from the
        instantaneous map at the moment of the wearable's last arrival, which
        can be many steps before whatever frame you're looking at. Neither is
        a bug -- confirmed by inspecting the actual target-selection code and
        plot indexing, no row/col mismatch -- but it means a wearable's route
        can legitimately look like it's not heading toward the visually
        reddest area in a given frame. Expected to matter more once the world
        is temporally dynamic: a target committed to at time T can go stale
        before the wearable arrives if the field moves underneath it, a real
        tension in commit-then-reassess (see wearable_policy="uncertainty_guided"
        docstring) worth addressing explicitly in that stage (e.g. capping
        transit distance per commit, or re-evaluating if the map has moved
        too much mid-transit) rather than a defect in this static-world test."""
        acc = np.zeros((self.grid_size, self.grid_size))
        cnt = np.zeros((self.grid_size, self.grid_size))
        for node in self.nodes.values():
            for (r, c), (_, nu, _, _) in node.cell_beliefs.items():
                acc[r, c] += 1.0 / max(nu, 1e-6)
                cnt[r, c] += 1
        with np.errstate(invalid="ignore"):
            out = acc / cnt
        out[cnt == 0] = np.nan
        return out

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
