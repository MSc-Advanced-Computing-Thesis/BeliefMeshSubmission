"""Tests for beliefmesh.node.node.Node."""

from __future__ import annotations

import torch

from beliefmesh.node.node import Node


def _images(n=4):
    torch.manual_seed(0)
    return torch.rand(n, 3, 28, 28)


def test_predict_returns_full_uncollapsed_tuple():
    node = Node(lr=3e-4, device=torch.device("cpu"))
    out = node.predict(_images())
    assert len(out) == 4  # (gamma, nu, alpha, beta) -- never a collapsed point estimate
    gamma, nu, alpha, beta = out
    assert gamma.shape == (4,)
    assert (nu > 0).all() and (alpha > 1).all() and (beta > 0).all()


def test_predict_does_not_track_gradients():
    node = Node(lr=3e-4, device=torch.device("cpu"))
    gamma, *_ = node.predict(_images())
    assert not gamma.requires_grad


def test_train_step_updates_parameters_and_returns_finite_loss():
    node = Node(lr=1e-3, device=torch.device("cpu"))
    before = [p.clone() for p in node.model.parameters()]
    targets = torch.rand(4) * 2 - 1
    loss = node.train_step(_images(), targets)
    assert isinstance(loss, float)
    assert torch.isfinite(torch.tensor(loss))
    changed = any(not torch.equal(b, p) for b, p in zip(before, node.model.parameters()))
    assert changed


def test_train_step_accepts_peer_gamma_as_pseudo_label():
    teacher = Node(lr=3e-4, device=torch.device("cpu"))
    student = Node(lr=3e-4, device=torch.device("cpu"))
    imgs = _images()
    gamma, *_ = teacher.predict(imgs)
    loss = student.train_step(imgs, gamma)  # Spec Sec 5: scalar pseudo-label, ordinary loss
    assert torch.isfinite(torch.tensor(loss))
