import numpy as np
import numpy.typing as npt


def _complete_cases(*args):
    mask = np.zeros(args[0].size, dtype=np.bool)
    for i in args:
        if i is not None:
            mask |= np.isnan(i)
    return ~mask

def _rdiff(a: np.ndarray, b: float, r: float) -> np.ndarray:
    if r == 0.0:
        with np.errstate(divide="ignore"):
            return np.log(a / b)
    elif r == 1.0:
        return a - b
    else:
        with np.errstate(divide="ignore"):
            return (a**r - b**r) / r

def _extended_mean_pow(x: np.ndarray, m: float, r: float, s: float):
    with np.errstate(divide="ignore"):
        res = _rdiff(x, m, r) / _rdiff(x, m, s)
    return np.where(np.isclose(x, m), m**(r - s), res)

def _pair_means(
    x: npt.ArrayLike,
    weights: tuple[npt.ArrayLike | None, npt.ArrayLike | None],
    order: tuple[float, float],
    drop_na: bool
) -> tuple[float, float]:
    x = np.asarray(x).ravel()
    w1 = np.asarray(weights[0]).ravel() if weights[0] is not None else None
    w2 = np.asarray(weights[1]).ravel() if weights[1] is not None else None
    if w1 is not None:
        if len(x)  != len(w1):
            raise ValueError("'x' and 'weights[0]' must be the same length")
    if w2 is not None:
        if len(x) != len(w2):
            raise ValueError("'x' and 'weights[1]' must be the same length")
    if drop_na:
        keep = _complete_cases(x, w1, w2)
        x = x[keep]
        if w1 is not None:
            w1 = w1[keep]
        if w2 is not None:
            w2 = w2[keep]
    return (mean(x, w1, order=order[0]), mean(x, w2, order=order[1]))


def mean(
    x: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    order: float = 1.0,
    drop_na: bool = False
) -> np.ndarray:
    x = np.asarray(x).ravel()
    if weights is not None:
        weights = np.asarray(weights).ravel()
        if len(x) != len(weights):
            raise ValueError("'x' and 'weights' must be the same length")
    order = float(order)
    if drop_na:
        keep = _complete_cases(x, weights)
        x = x[keep]
        if weights is not None:
            weights = weights[keep]
    if weights is None:
        with np.errstate(divide="ignore"):
            if order == 0.0:
                return np.exp(np.average(np.log(x)))
            else:
                return np.average(x**order)**(1/order)
    else:
        with np.errstate(divide="ignore"):
            if order == 0.0:
                return np.exp(np.average(np.log(x), weights=weights))
            else:
                return np.average(x**order, weights=weights)**(1/order)    

def nested_mean(
    x: npt.ArrayLike,
    weights: tuple[npt.ArrayLike | None, npt.ArrayLike | None] = (None, None),
    *,
    order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    outer_weights: npt.ArrayLike = np.array([1, 1]),
    drop_na: bool = False
) -> np.ndarray:
    m = _pair_means(x, weights, order[1], drop_na)
    outer_weights = np.asarray(outer_weights).ravel()
    return mean(m, outer_weights, order=order[0], drop_na=drop_na)

def scale_weights(x: npt.ArrayLike) -> np.ndarray:
    return x / np.sum(x[~np.isnan(x)])

def transmute_weights(
    x: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    order: float = 0.0,
    to: float = 1.0,
    mean_value: float | None = None
) -> npt.ArrayLike:
    x = np.asarray(x).ravel()
    if weights is not None:
        weights = np.asarray(weights).ravel()
        if len(x) != len(weights):
            raise ValueError("'x' and 'weights' must be the same length")
    order = float(order)
    to = float(to)
    if order == to:
        if weights is None:
            weights = np.repeat(1, len(x))
        res = np.where(np.isnan(x), np.nan, weights)
    else:
        if mean_value is None:
            mean_value = mean(x, weights, order=order, drop_na=True)
        else:
            mean_value = float(mean_value)
        if weights is None:
            res = _extended_mean_pow(x, mean_value, order, to)
        else:
            res = weights * _extended_mean_pow(x, mean_value, order, to)
    return scale_weights(res)

def nested_transmute(
    x: npt.ArrayLike,
    weights: tuple[npt.ArrayLike | None, npt.ArrayLike | None] = (None, None),
    *,
    order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    outer_weights: npt.ArrayLike = np.array([1, 1]),
    to: float = 1.0,
    pivot: float = 0.0
) -> npt.ArrayLike:
    x = np.asarray(x).ravel()
    w1 = np.asarray(weights[0]).ravel() if weights[0] is not None else None
    w2 = np.asarray(weights[1]).ravel() if weights[1] is not None else None
    if w1 is not None:
        if len(x) != len(w1):
            raise ValueError("'x' and 'weights[0]' must be the same length")
    if w2 is not None:
        if len(x) != len(w2):
            raise ValueError("'x' and 'weights[1]' must be the same length")
    to_na = _complete_cases(x, w1, w2)
    x = np.where(to_na, x, np.nan)
    m = _pair_means(x, (w1, w2), order[1], drop_na=True)
    v1 = transmute_weights(x, w1, order=order[1][0], to=pivot, mean_value=m[0])
    v2 = transmute_weights(x, w2, order=order[1][1], to=pivot, mean_value=m[1])
    t = transmute_weights(m, outer_weights, order=order[0], to=pivot)
    if np.isnan(t[0]):
        return transmute_weights(x, v2*t[1], order=pivot, to=to)
    elif np.isnan(t[1]):
        return transmute_weights(x, v1*t[0], order=pivot, to=to)
    else:
        return transmute_weights(x, v1*t[0] + v2*t[1], order=pivot, to=to)

def factor_weights(
    x: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    order: float = 1.0
) -> np.ndarray:
    x = np.asarray(x).ravel()
    if weights is not None:
        weights = np.asarray(weights).ravel()
        if len(x) != len(weights):
            raise ValueError("'x' and 'weights' must be the same length")
    order = float(order)
    if order == 0.0:
        if weights is None:
            weights = np.repeat(1, len(x))
        res = np.where(np.isnan(x), np.nan, weights)
    else:
        if weights is None:
            res = x**order
        else:
            res = weights * x**order
    return scale_weights(res)