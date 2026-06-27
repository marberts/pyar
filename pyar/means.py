import math
import numpy as np
import numpy.typing as npt


def _complete_cases(*args):
    mask = np.zeros(args[0].size, dtype=np.bool)
    for i in args:
        if i is not None:
            mask |= np.isnan(i)
    return ~mask


def _rdiff(a: np.ndarray, b: float, r: float) -> np.ndarray:
    with np.errstate(divide="ignore", invalid="ignore"):
        if r == 0.0:
            return np.log(a / b)
        elif r == 1.0:
            return a - b
        else:
            return (a**r - b**r) / r


def _extended_mean_pow(x: np.ndarray, m: float, r: float, s: float):
    with np.errstate(divide="ignore", invalid="ignore"):
        res = _rdiff(x, m, r) / _rdiff(x, m, s)
        return np.where(np.isclose(x, m), m ** (r - s), res)


def _pair_means(
    x: npt.ArrayLike,
    weights: tuple[npt.ArrayLike | None, npt.ArrayLike | None],
    order: tuple[float, float],
    skipnan: bool,
) -> tuple[float, float]:
    x = np.asarray(x).ravel()
    w1 = np.asarray(weights[0]).ravel() if weights[0] is not None else None
    w2 = np.asarray(weights[1]).ravel() if weights[1] is not None else None
    if w1 is not None:
        if len(x) != len(w1):
            raise ValueError("`x` and `weights[0]` must be the same length")
    if w2 is not None:
        if len(x) != len(w2):
            raise ValueError("`x` and `weights[1]` must be the same length")
    if skipnan:
        keep = _complete_cases(x, w1, w2)
        x = x[keep]
        if w1 is not None:
            w1 = w1[keep]
        if w2 is not None:
            w2 = w2[keep]
    return mean(x, w1, order=order[0]), mean(x, w2, order=order[1])


def mean(
    x: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    order: float = 1.0,
    skipnan: bool = False,
) -> float:
    """Generalized mean

    Compute the weighted generalized mean of a flat array of values.

    Parameters
    ----------
    x : ArrayLike
        A flat array of positive values.
    weights : ArrayLike | None, optional
              A flat array of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : float, optional
            The order the generalized mean, defaulting to an arithmetic mean.
    skipnan : bool, optional
              Drop NaNs in `x` and `weights`. By default these values are not
              dropped.

    Returns
    -------
    float
        The weighted generalized mean.

    Notes
    -----
    The generalized mean can be defined on the extended real line, so that
    `order` = -Inf / Inf returns `min(x)`/`max(x)`, to agree with the definition
    by Bullen (2003). This is not implemented, and order must be finite.

    The convention by Hardy et al. (1952, p. 13) is used in cases where `x` has
    zeros: the generalized mean is 0 whenever `weights` is strictly positive and
    `order < 0`. The analogous convention holds whenever at least one element of
    `x` is Inf: the generalized mean is Inf whenever `weights` is strictly
    positive and `order > 0`.

    References
    ----------
    Bullen, P. S. (2003). Handbook of Means and Their Inequalities. Springer
    Science+Business Media.

    Hardy, G., Littlewood, J. E., and Polya, G. (1952). Inequalities
    (2nd edition). Cambridge University Press.

    Examples
    --------
    """
    x = np.asarray(x).ravel()
    if weights is not None:
        weights = np.asarray(weights).ravel()
        if len(x) != len(weights):
            raise ValueError("`x` and `weights` must be the same length")
    order = float(order)
    if not math.isfinite(order):
        raise ValueError("`order` must be finite")
    if skipnan:
        keep = _complete_cases(x, weights)
        x = x[keep]
        if weights is not None:
            weights = weights[keep]
    with np.errstate(divide="ignore", invalid="ignore"):
        if weights is None:
            if order == 0.0:
                return np.exp(np.average(np.log(x)))
            else:
                return np.average(x**order) ** (1 / order)
        else:
            if order == 0.0:
                return np.exp(np.average(np.log(x), weights=weights))
            else:
                return np.average(x**order, weights=weights) ** (1 / order)


def nested_mean(
    x: npt.ArrayLike,
    weights: tuple[npt.ArrayLike | None, npt.ArrayLike | None] = (None, None),
    *,
    order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    outer_weights: npt.ArrayLike = np.array([1, 1]),
    skipnan: bool = False,
) -> float:
    """Nested generalized means

    Compute the generalized mean of a pair of generalized means
    (i.e., crossing means).

    Parameters
    ----------
    x : ArrayLike
        A flat array of positive values.
    weights : tuple[ArrayLike | None, ArrayLike | None], optional
              A pair of flat arrays of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : tuple[float, tuple[float, float]], optional
            The order the outer and inner generalized means. The default
            computes the geometric mean of the arithmetic and harmonic means.
    outer_weights: ArrayLike, optional
                   A pair of weights for the outer generalized mean. The
                   defaults weights each of the inner generalized means
                   equally.
    skipnan : bool, optional
              Drop NaNs in `x` and `weights`. By default these values are not
              dropped.

    Returns
    -------
    float
        The nested generalized mean, calculated as
        mean([
              mean(x, weights[0], order=order[1][0]),
              mean(x, weights[1], order=order[1][1])
             ],
            outer_weights,
            order=order[0]
        ).

    Notes
    -----
    Removal of NaNs is balanced across all inputs.

    Examples
    --------
    """
    m = _pair_means(x, weights, order[1], skipnan)
    outer_weights = np.asarray(outer_weights).ravel()
    return mean(m, outer_weights, order=order[0], skipnan=skipnan)


def scale_weights(x: npt.ArrayLike) -> np.ndarray:
    """Scale weights

    Scale a flat array of weights so that it sums to one, ignoring NaNs.

    Parameters
    ----------
    x : ArrayLike
        A flat array of weights.

    Returns
    -------
    ndarray
        A copy of `x`, normalized to sum to one.

    Examples
    --------
    """
    total = np.sum(x[~np.isnan(x)])
    if total <= 0.0:
        raise ValueError("cannot scale weights to sum to one")
    return x / total


def transmute_weights(
    x: npt.ArrayLike,
    weights: npt.ArrayLike | None = None,
    *,
    order: float = 0.0,
    to: float = 1.0,
    mean_value: float | None = None,
) -> npt.ArrayLike:
    """Transmute weights in a generalized mean

    Transmute the weights to turn a generalized mean of a given order into a
    generalized mean of any other order. That is, derive weights `v` such that
    `mean(x, wights, order=order) == mean(x, v, order=to)`.

    Parameters
    ----------
    x : ArrayLike
        A flat array of positive values.
    weights : ArrayLike | None, optional
              A flat array of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : float, optional
            The order the generalized mean, defaulting to a geometric mean.
    to : float, optional
         The target order of the generalized mean for the transmuted weights.
         The default is an arithmetic mean.
    mean_value : float, optional.
         The value of the generalized mean of `x`. Can be provided if known to
         save recomputing it.

    Returns
    -------
    ndarray
        An array of weights, the same length as `x`.

    References
    ----------
    Balk, B. M. 2008. Price and Quantity Index Numbers. Cambridge University
    Press.

    Martin, S. 2021. A Note on Generalized Decompositions for Price Indexes.
    Prices Analytical Series. Statistics Canada Catalogue 62F0014M.

    Examples
    --------
    """
    x = np.asarray(x).ravel()
    if weights is not None:
        weights = np.asarray(weights).ravel()
        if len(x) != len(weights):
            raise ValueError("`x` and `weights` must be the same length")
    order = float(order)
    if not math.isfinite(order):
        raise ValueError("`order` must be finite")
    to = float(to)
    if not math.isfinite(to):
        raise ValueError("`to` must be finite")
    if order == to:
        if weights is None:
            weights = np.repeat(1.0, len(x))
        res = np.where(np.isnan(x), np.nan, weights)
    else:
        if mean_value is None:
            mean_value = mean(x, weights, order=order, skipnan=True)
        else:
            mean_value = float(mean_value)
        if weights is None:
            res = _extended_mean_pow(x, mean_value, order, to)
        else:
            res = weights * _extended_mean_pow(x, mean_value, order, to)
    return res


def nested_transmute(
    x: npt.ArrayLike,
    weights: tuple[npt.ArrayLike | None, npt.ArrayLike | None] = (None, None),
    *,
    order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    outer_weights: npt.ArrayLike = np.array([1, 1]),
    to: float = 1.0,
    pivot: float = None,
) -> npt.ArrayLike:
    """Transmute weights in a nested generalized mean

    Transmute the weights to turn a nested generalized mean of a given order
    into a generalized mean of any other order. That is, derive weights `v` such
    that `nested_mean(x, weights, order=order) == mean(x, v, order=to)`.

    Parameters
    ----------
    x : ArrayLike
        A flat array of positive values.
    weights : tuple[ArrayLike | None, ArrayLike | None], optional
              A pair of flat arrays of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : tuple[float, tuple[float, float]], optional
            The order the outer and inner generalized means. The default
            computes the geometric mean of the arithmetic and harmonic means.
    outer_weights: ArrayLike, optional
                   A pair of weights for the outer generalized mean. The
                   default weights each of the inner generalized means
                   equally.
    to : float, optional
         The target order of the generalized mean for the transmuted weights.
         The default is an arithmetic mean.
    pivot : float, optional
         The pivot value used to transmute the weights. The default uses
         `order[0]`.

    Returns
    -------
    ndarray
        An array of weights, the same length as `x`.

    References
    ----------
    Balk, B. M. 2008. Price and Quantity Index Numbers. Cambridge University
    Press.

    Martin, S. 2021. A Note on Generalized Decompositions for Price Indexes.
    Prices Analytical Series. Statistics Canada Catalogue 62F0014M.

    Examples
    --------
    """
    x = np.asarray(x).ravel()
    w1 = np.asarray(weights[0]).ravel() if weights[0] is not None else None
    w2 = np.asarray(weights[1]).ravel() if weights[1] is not None else None
    if w1 is not None:
        if len(x) != len(w1):
            raise ValueError("`x` and `weights[0]` must be the same length")
    if w2 is not None:
        if len(x) != len(w2):
            raise ValueError("`x` and `weights[1]` must be the same length")
    if pivot is None:
        pivot = order[0]
    to_na = _complete_cases(x, w1, w2)
    x = np.where(to_na, x, np.nan)
    m = _pair_means(x, (w1, w2), order[1], skipnan=True)
    v1 = transmute_weights(x, w1, order=order[1][0], to=pivot, mean_value=m[0])
    v2 = transmute_weights(x, w2, order=order[1][1], to=pivot, mean_value=m[1])
    t = transmute_weights(m, outer_weights, order=order[0], to=pivot)
    if np.isnan(t[0]):
        return transmute_weights(x, v2 * t[1], order=pivot, to=to)
    elif np.isnan(t[1]):
        return transmute_weights(x, v1 * t[0], order=pivot, to=to)
    else:
        return transmute_weights(x, v1 * t[0] + v2 * t[1], order=pivot, to=to)


def factor_weights(
    x: npt.ArrayLike, weights: npt.ArrayLike | None = None, *, order: float = 1.0
) -> np.ndarray:
    """Factor weights

    Factor the weights to turn generalized mean of products into the product
    of generalized means.

    Parameters
    ----------
    x : ArrayLike
        A flat array of positive values.
    weights : ArrayLike | None, optional
              A flat array of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : float, optional
            The order the generalized mean, defaulting to an arithmetic mean.

    Returns
    -------
    ndarray
        An array of weights, the same length as `x`.

    Examples
    --------
    """
    x = np.asarray(x).ravel()
    if weights is not None:
        weights = np.asarray(weights).ravel()
        if len(x) != len(weights):
            raise ValueError("`x` and `weights` must be the same length")
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
