"""
Selection related metrics used across LibMoE v2 analysis scripts.
"""

from __future__ import annotations

from typing import Any, Iterable, List, Sequence

import numpy as np


TokenSelection = Sequence[Sequence[Any]]


def average_switch_fraction(epoch_a: TokenSelection, epoch_b: TokenSelection) -> float:
    """
    Fraction of selected expert slots that change between two epochs.
    """
    epoch_a = _normalize_epoch(epoch_a)
    epoch_b = _normalize_epoch(epoch_b)
    fractions: List[float] = []
    for token_a, token_b in zip(epoch_a, epoch_b):
        if not token_a:
            continue
        switches = sum(a != b for a, b in zip(token_a, token_b))
        fractions.append(switches / len(token_a))
    return _safe_average(fractions)


def average_overlap_fraction(epoch_a: TokenSelection, epoch_b: TokenSelection) -> float:
    """
    Fraction of overlapping experts regardless of position.
    """
    epoch_a = _normalize_epoch(epoch_a)
    epoch_b = _normalize_epoch(epoch_b)
    fractions: List[float] = []
    for token_a, token_b in zip(epoch_a, epoch_b):
        if not token_a:
            continue
        overlap = sum(expert in token_b for expert in token_a)
        fractions.append(overlap / len(token_a))
    return _safe_average(fractions)


def average_position_match(epoch_a: TokenSelection, epoch_b: TokenSelection) -> float:
    """
    Fraction of experts that remain in the same slot between two epochs.
    """
    epoch_a = _normalize_epoch(epoch_a)
    epoch_b = _normalize_epoch(epoch_b)
    matches: List[float] = []
    for token_a, token_b in zip(epoch_a, epoch_b):
        if not token_a:
            continue
        equal_slots = sum(token_a[idx] == token_b[idx] for idx in range(len(token_a)))
        matches.append(equal_slots / len(token_a))
    return _safe_average(matches)


def _safe_average(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return float("nan")
    return float(np.average(values))


def _normalize_epoch(epoch: Iterable[Any]) -> List[List[float]]:
    normalized: List[List[float]] = []
    for token in epoch:
        if isinstance(token, Sequence) and not isinstance(token, (str, bytes)):
            normalized.append([float(element) for element in token])
        else:
            normalized.append([float(token)])
    return normalized
