import math
import pandas as pd
import numpy as np


def _complete_cases(*args: pd.Series, invert: bool = False) -> np.ndarray:
    mask = np.zeros(args[0].size, dtype=np.bool)
    for i in args:
        if i is not None:
            mask |= i.isna()
    return mask if invert else ~mask


def _balance_nas(*args: pd.Series) -> list[pd.Series]:
    to_na = _complete_cases(*args)
    if np.any(to_na):
        return [x.where(to_na) for x in args]
    else:
        return args


def _rdiff(x: pd.Series, y: float, r: float) -> pd.Series:
    if r == 0.0:
        return np.log(x / y)
    elif r == 1.0:
        return x - y
    else:
        return (x**r - y**r) / r


def _extended_mean_pow(x: pd.Series, m: float, r: float, s: float) -> pd.Series:
    res = _rdiff(x, m, r) / _rdiff(x, m, s)
    return res.where(~np.isclose(x, m), m ** (r - s))


def _pair_means(
    x: pd.Series,
    weights: tuple[pd.Series | None, pd.Series | None],
    order: tuple[float, float],
    skipna: bool,
) -> tuple[float, float]:
    w1, w2 = weights
    if w1 is not None:
        if len(x) != len(w1):
            raise ValueError("`x` and `weights[0]` must be the same length")
    if w2 is not None:
        if len(x) != len(w2):
            raise ValueError("`x` and `weights[1]` must be the same length")
    if skipna:
        keep = _complete_cases(x, w1, w2)
        x = x[keep]
        if w1 is not None:
            w1 = w1[keep]
        if w2 is not None:
            w2 = w2[keep]
    return mean(x, w1, order=order[0]), mean(x, w2, order=order[1])


def mean(
    x: pd.Series,
    weights: pd.Series | None = None,
    *,
    order: float = 1.0,
    skipna: bool = False,
) -> float:
    """Generalized mean

    Compute the weighted generalized mean of a flat array of values.

    Parameters
    ----------
    x : Series
        A Series of positive values.
    weights : Series | None, optional
              A Series of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : float, optional
            The order the generalized mean, defaulting to an arithmetic mean.
    skipna : bool, optional
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
    if weights is not None:
        if len(x) != len(weights):
            raise ValueError("`x` and `weights` must be the same length")
    order = float(order)
    if not math.isfinite(order):
        raise ValueError("`order` must be finite")
    if skipna:
        keep = _complete_cases(x, weights)
        x = x[keep]
        if weights is not None:
            weights = weights[keep]
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
    x: pd.Series,
    weights: tuple[pd.Series | None, pd.Series | None] = (None, None),
    *,
    order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    outer_weights: tuple[float, float] = (1.0, 1.0),
    skipna: bool = False,
) -> float:
    """Nested generalized means

    Compute the generalized mean of a pair of generalized means
    (i.e., crossing means).

    Parameters
    ----------
    x : Series
        A Series of positive values.
    weights : tuple[Series | None, Series | None], optional
              A pair of Series of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : tuple[float, tuple[float, float]], optional
            The order the outer and inner generalized means. The default
            computes the geometric mean of the arithmetic and harmonic means.
    outer_weights: tuple[float, float], optional
                   A pair of weights for the outer generalized mean. The
                   defaults weights each of the inner generalized means
                   equally.
    skipna : bool, optional
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
    m = _pair_means(x, weights, order[1], skipna)
    return mean(m, pd.Series(outer_weights), order=order[0], skipna=skipna)


def extended_mean(
    x: pd.Series, y: pd.Series, order: tuple[float, float] = (0.0, 1.0)
) -> pd.Series:
    """Extended mean

    Calculated the pairwise extended mean to two arrays.

    Parameters
    ----------
    x : Series
        A Series of positive values.
    y : Series
        A Series of positive values.
    order : tuple[float, float], optional
            A pair of numbers giving the order of the extended mean. Defaults to
            computing a logarithmic mean.

    Returns
    -------
    Series
        The pairwise extended mean.

    References
    ----------
    Bullen, P. S. (2003). Handbook of Means and Their Inequalities.
    Springer Science+Business Media.

    Examples
    --------

    """
    if not all(math.isfinite(x) for x in order):
        raise ValueError("`order` must be a pair of finite numbers")
    r, s = order
    if r == 0.0 and s == 0.0:
        res = np.sqrt(x * y)
    elif r == 0.0:
        res = ((x**s - y**s) / np.log(x / y) / s) ** (1 / s)
    elif s == 0.0:
        res = ((x**r - y**r) / np.log(x / y) / r) ** (1 / r)
    elif r == s:
        res = np.exp((x**r * np.log(x) - y**r * np.log(y)) / (x**r - y**r) - 1 / r)
    else:
        res = ((x**s - y**s) / (x**r - y**r) * (r / s)) ** (1 / (s - r))
    # Set output to x when x == y.
    return x(np.isclose(x, y), res)


def scale_weights(x: pd.Series) -> pd.Series:
    """Scale weights

    Scale a Series of weights so that it sums to one, ignoring NaNs.

    Parameters
    ----------
    x : Series
        A Series of weights.

    Returns
    -------
    Series
        A copy of `x`, normalized to sum to one.

    Examples
    --------
    """
    total = x.sum()
    if total <= 0.0:
        raise ValueError("cannot scale weights to sum to one")
    return x / total


def transmute_weights(
    x: pd.Series,
    weights: pd.Series | None = None,
    *,
    order: float = 0.0,
    to: float = 1.0,
    mean_value: float | None = None,
) -> pd.Series:
    """Transmute weights in a generalized mean

    Transmute the weights to turn a generalized mean of a given order into a
    generalized mean of any other order. That is, derive weights `v` such that
    `mean(x, wights, order=order) == mean(x, v, order=to)`.

    Parameters
    ----------
    x : Series
        A Series of positive values.
    weights : Series | None, optional
              A fSeries of weights for the values in `x`. If None
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
    Series
        A Series of weights, the same length as `x`.

    References
    ----------
    Balk, B. M. 2008. Price and Quantity Index Numbers. Cambridge University
    Press.

    Martin, S. 2021. A Note on Generalized Decompositions for Price Indexes.
    Prices Analytical Series. Statistics Canada Catalogue 62F0014M.

    Examples
    --------
    """
    if weights is not None:
        if len(x) != len(weights):
            raise ValueError("`x` and `weights` must be the same length")
    order, to = float(order), float(to)
    if not math.isfinite(order):
        raise ValueError("`order` must be finite")
    if not math.isfinite(to):
        raise ValueError("`to` must be finite")
    if order == to:
        if weights is None:
            weights = pd.Series(np.repeat(1.0, len(x)))
        res = weights.where(x.notna())
    else:
        if mean_value is None:
            mean_value = mean(x, weights, order=order, skipna=True)
        else:
            mean_value = float(mean_value)
        if weights is None:
            res = _extended_mean_pow(x, mean_value, order, to)
        else:
            res = weights * _extended_mean_pow(x, mean_value, order, to)
    return scale_weights(res)


def nested_transmute(
    x: pd.Series,
    weights: tuple[pd.Series | None, pd.Series | None] = (None, None),
    *,
    order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    outer_weights: tuple[float, float] = (1.0, 1.0),
    to: float = 1.0,
    pivot: float = None,
) -> pd.Series:
    """Transmute weights in a nested generalized mean

    Transmute the weights to turn a nested generalized mean of a given order
    into a generalized mean of any other order. That is, derive weights `v` such
    that `nested_mean(x, weights, order=order) == mean(x, v, order=to)`.

    Parameters
    ----------
    x : Series
        A Series of positive values.
    weights : tuple[ArrayLike | None, ArrayLike | None], optional
              A pair of flat arrays of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : tuple[float, tuple[float, float]], optional
            The order the outer and inner generalized means. The default
            computes the geometric mean of the arithmetic and harmonic means.
    outer_weights: tuple[float, float], optional
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
    Series
        A Series of weights, the same length as `x`.

    References
    ----------
    Balk, B. M. 2008. Price and Quantity Index Numbers. Cambridge University
    Press.

    Martin, S. 2021. A Note on Generalized Decompositions for Price Indexes.
    Prices Analytical Series. Statistics Canada Catalogue 62F0014M.

    Examples
    --------
    """
    w1, w2 = weights
    if w1 is not None:
        if len(x) != len(w1):
            raise ValueError("`x` and `weights[0]` must be the same length")
    if w2 is not None:
        if len(x) != len(w2):
            raise ValueError("`x` and `weights[1]` must be the same length")
    if pivot is None:
        pivot = order[0]
    to_na = _complete_cases(x, w1, w2)
    x = x.where(to_na)
    m = _pair_means(x, (w1, w2), order[1], skipna=True)
    v1 = transmute_weights(x, w1, order=order[1][0], to=pivot, mean_value=m[0])
    v2 = transmute_weights(x, w2, order=order[1][1], to=pivot, mean_value=m[1])
    t = transmute_weights(
        pd.Series(m), pd.Series(outer_weights), order=order[0], to=pivot
    )
    if math.isnan(t[0]):
        return transmute_weights(x, v2 * t[1], order=pivot, to=to)
    elif math.isnan(t[1]):
        return transmute_weights(x, v1 * t[0], order=pivot, to=to)
    else:
        return transmute_weights(x, v1 * t[0] + v2 * t[1], order=pivot, to=to)


def factor_weights(
    x: pd.Series, weights: pd.Series | None = None, *, order: float = 1.0
) -> pd.Series:
    """Factor weights

    Factor the weights to turn generalized mean of products into the product
    of generalized means.

    Parameters
    ----------
    x : Series
        A Series of positive values.
    weights : Series | None, optional
              A Series of weights for the values in `x`. If None
              (the default) then each element of `x` gets equal weight.
    order : float, optional
            The order the generalized mean, defaulting to an arithmetic mean.

    Returns
    -------
    Series
        A Series of weights, the same length as `x`.

    Examples
    --------
    """
    if weights is not None:
        if len(x) != len(weights):
            raise ValueError("`x` and `weights` must be the same length")
    order = float(order)
    if order == 0.0:
        if weights is None:
            weights = pd.Series(np.repeat(1, len(x)))
        res = weights.where(x.notna())
    else:
        if weights is None:
            res = x**order
        else:
            res = weights * x**order
    return scale_weights(res)
