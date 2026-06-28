import numpy as np
import numpy.typing as npt
from .means import (
    mean,
    scale_weights,
    _balance_nans,
    transmute_weights,
    nested_transmute,
    flatten_inputs,
)


@flatten_inputs
def hallerbach(p1, p0, v1, v0):
    p1, p0, v1, v0 = _balance_nans(p1, p0, v1, v0)
    r = p1 / p0
    v0, v1 = scale_weights(v0), scale_weights(v1 / r)
    laspeyres, paasche = mean(r, v0, skipnan=True), mean(r, v1, skipnan=True)
    return (
        0.5 * np.sqrt(paasche / laspeyres) * v0
        + 0.5 * np.sqrt(laspeyres / paasche) * v1
    ) * r


@flatten_inputs
def diewert(p1, p0, v1, v0):
    p1, p0, v1, v0 = _balance_nans(p1, p0, v1, v0)
    r = p1 / p0
    laspeyres, paasche = (
        mean(r, v0, skipnan=True),
        mean(r, v1, order=-1.0, skipnan=True),
    )
    fisher = mean((laspeyres, paasche), order=0.0, skipnan=True)
    wl, wp = scale_weights(v0), scale_weights(v1 / r)
    return (wl + laspeyres * wp) / (1 + fisher) * (r - 1)


@flatten_inputs
def vonBortkiewicz(p1, p0, v1, v0, *, type: str = "arithmetic"):
    p1, p0, v1, v0 = _balance_nans(p1, p0, v1, v0)
    if type not in ["arithmetic", "geometric"]:
        raise ValueError("`type` must be one of 'arithmetic' or 'geometric'")
    if type == "arithmetic":
        q1, q0 = v1 / p1, v0 / p0
        price, quantity = (
            mean(p1 / p0, v0, skipnan=True),
            mean(q1 / q0, v1, skipnan=True),
        )
        return scale_weights(v0) * (p1 / p0 / price - 1) * (q1 / q0 / quantity - 1)
    elif type == "geometric":
        v1, v0 = scale_weights(v1), scale_weights(v0)
        r = p1 / p0
        return v0 * (v1 / v0 - 1) * np.log(r / mean(r, v0, order=0, skipnan=True))


def contributions(x: npt.ArrayLike, weights=npt.ArrayLike, order: float = 0.0):
    weights = transmute_weights(x, weights, order=order, to=1.0)
    return weights * (x - 1)


def nested_contributions(
    x: npt.ArrayLike,
    weights: tuple[npt.ArrayLike | None, npt.ArrayLike | None] = (None, None),
    *,
    order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    outer_weights: npt.ArrayLike = np.array([1, 1]),
    pivot: float = None,
) -> np.ndarray:
    weights = nested_transmute(
        x, weights, order=order, outer_weights=outer_weights, pivot=pivot
    )
    return weights * (x - 1)
