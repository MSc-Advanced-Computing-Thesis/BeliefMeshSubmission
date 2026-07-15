"""Tests for beliefmesh.data (filters, digit-7 dataset, held-out split).

The split determinism tests matter most: every stage's evaluation integrity
rests on the seed-42 held-out indices being stable across runs and disjoint
from training (Spec Sec 2).
"""

from __future__ import annotations

import numpy as np
import torch

from beliefmesh.data.digits import RotatedDigitDataset, get_digit7_splits
from beliefmesh.data.filters import apply_colour_filter, make_green_digits


# --- filters ---

def test_neutral_filter_is_noop():
    img = torch.rand(3, 28, 28)
    assert torch.equal(apply_colour_filter(img, "neutral", 1.0), img)


def test_red_and_blue_suppress_green_digit_symmetrically():
    # Spec Sec 2: the digit is green so red and blue filters of equal strength
    # suppress it equally -- no directional bias between the two shifts.
    img = torch.zeros(3, 4, 4)
    img[1] = 0.8  # pure green everywhere, like digit pixels
    red = apply_colour_filter(img, "red", 1.0)
    blue = apply_colour_filter(img, "blue", 1.0)
    torch.testing.assert_close(red[1], blue[1])  # green channel equally suppressed
    torch.testing.assert_close(red.sum(), blue.sum())


def test_filter_strength_zero_is_noop():
    img = torch.rand(3, 28, 28)
    torch.testing.assert_close(apply_colour_filter(img, "red", 0.0), img)


def test_filter_batch_matches_single():
    imgs = torch.rand(4, 3, 8, 8)
    batched = apply_colour_filter(imgs, "red", 0.7)
    singles = torch.stack([apply_colour_filter(im, "red", 0.7) for im in imgs])
    torch.testing.assert_close(batched, singles)


def test_make_green_digits_recolours_dark_pixels_only():
    img = torch.ones(3, 4, 4)      # white background
    img[:, 1, 1] = 0.0             # one dark (digit) pixel
    out = make_green_digits(img)
    torch.testing.assert_close(out[:, 1, 1], torch.tensor([0.0, 0.8, 0.0]))
    assert torch.equal(out[:, 0, 0], torch.ones(3))  # background untouched


# --- splits ---

def test_split_deterministic_across_calls():
    t1, e1 = get_digit7_splits()
    t2, e2 = get_digit7_splits()
    assert np.array_equal(t1, t2)
    assert np.array_equal(e1, e2)


def test_split_disjoint_and_complete():
    train, evl = get_digit7_splits()
    assert len(np.intersect1d(train, evl)) == 0
    n = len(train) + len(evl)
    assert set(np.concatenate([train, evl]).tolist()) == set(range(n))


def test_split_proportions():
    train, evl = get_digit7_splits(eval_fraction=0.2)
    total = len(train) + len(evl)
    assert abs(len(evl) / total - 0.2) < 0.01


def test_different_seed_gives_different_split():
    _, e42 = get_digit7_splits(seed=42)
    _, e43 = get_digit7_splits(seed=43)
    assert not np.array_equal(e42, e43)


# --- dataset ---

def test_dataset_item_shape_and_label_range():
    train_idx, _ = get_digit7_splits()
    ds = RotatedDigitDataset(subset_indices=train_idx[:8])
    assert len(ds) == 8
    img, label = ds[0]
    assert img.shape == (3, 28, 28)
    assert img.dtype == torch.float32
    assert 0.0 <= img.min() and img.max() <= 1.0
    assert -1.0 <= label.item() < 1.0


def test_dataset_digit_is_green_background_white():
    ds = RotatedDigitDataset(angle_range=(0.0, 0.0))  # unrotated for a clean check
    img, _ = ds[0]
    digit_mask = img[1] > img[0]  # green channel dominates on digit pixels
    assert digit_mask.any()
    background = img[:, 0, 0]  # corner is background: white
    torch.testing.assert_close(background, torch.ones(3))
