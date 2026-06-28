import numpy as np
import numpy.typing as npt
from .means import (
    mean,
    nested_mean,
    _balance_nans,
    scale_weights,
    flatten_inputs,
    extended_mean,
)


@flatten_inputs
def carli(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, order=1.0, skipnan=skipnan)


@flatten_inputs
def dutot(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, p0, order=1.0, skipnan=skipnan)


@flatten_inputs
def jevons(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, order=0.0, skipnan=skipnan)


@flatten_inputs
def cswd(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return nested_mean(p1 / p0, order=(0.0, (1.0, -1.0)), skipnan=skipnan)


@flatten_inputs
def balk_walsh(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return nested_mean(p1 / p0, order=(0.0, (0.5, 0.5)), skipnan=skipnan)


@flatten_inputs
def hybrid_cswd(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, np.sqrt(p0 / p1), order=1.0, skipnan=skipnan)


@flatten_inputs
def coggshall(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, order=-1.0, skipnan=skipnan)


@flatten_inputs
def laspeyres(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q0: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p0 * q0, order=1.0, skipnan=skipnan)


@flatten_inputs
def palgrave(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q1: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p1 * q1, order=1.0, skipnan=skipnan)


@flatten_inputs
def paasche(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q1: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p1 * q1, order=-1.0, skipnan=skipnan)


@flatten_inputs
def geo_laspeyres(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q0: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p0 * q0, order=0.0, skipnan=skipnan)


@flatten_inputs
def geo_paasche(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q1: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p1 * q1, order=0.0, skipnan=skipnan)


@flatten_inputs
def fisher(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return nested_mean(
        p1 / p0, (p0 * q0, p1 * q1), order=(0.0, (1.0, -1.0)), skipnan=skipnan
    )


@flatten_inputs
def tornqvist(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return nested_mean(
        p1 / p0, (p0 * q0, p1 * q1), order=(0.0, (0.0, 0.0)), skipnan=skipnan
    )


@flatten_inputs
def drobisch(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return nested_mean(
        p1 / p0, (p0 * q0, p1 * q1), order=(1.0, (1.0, -1.0)), skipnan=skipnan
    )


@flatten_inputs
def walsh1(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return mean(p1 / p0, p0 * np.sqrt(q0 * q1), order=1.0, skipnan=skipnan)


@flatten_inputs
def marshall_edgeworth(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return mean(p1 / p0, p0 * (q0 + q1), order=1.0, skipnan=skipnan)


@flatten_inputs
def geary_khamis(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return mean(p1 / p0, p0 * (1 / q0 + 1 / q1), order=1.0, skipnan=skipnan)


@flatten_inputs
def walsh2(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return mean(p1 / p0, np.sqrt(p1 * q1 * p0 * q0), order=0.0, skipnan=skipnan)


@flatten_inputs
def theil(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    if skipnan:
        p1, p0, q1, q0 = _balance_nans(p1, p0, q1, q0)
    w1 = scale_weights(p1 * q1)
    w0 = scale_weights(p0 * q0)
    return mean(p1 / p0, (w1 * w0 * (w1 + w0) / 2) ** (1 / 3), order=0.0)


@flatten_inputs
def rao(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    if skipnan:
        p1, p0, q1, q0 = _balance_nans(p1, p0, q1, q0)
    w1 = scale_weights(p1 * q1)
    w0 = scale_weights(p0 * q0)
    return mean(p1 / p0, w1 * w0 / (w1 + w0) / 2, order=0.0)


@flatten_inputs
def sato_vartia(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan: bool = False,
) -> np.float64:
    return mean(
        p1 / p0,
        extended_mean(p0 * q0 / np.sum(p0 * q0), p1 * q1 / np.sum(p1 * q1)),
        order=0.0,
        skipnan=skipnan,
    )


@flatten_inputs
def lloyd_moulton(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q0: npt.ArrayLike,
    elasticity: float,
    skipnan: bool = False,
) -> np.float64:
    return mean(p1 / p0, p0 * q0, order=elasticity, skipnan=skipnan)


@flatten_inputs
def lehr(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan: bool = False,
) -> np.float64:
    if skipnan:
        p1, p0, q1, q0 = _balance_nans(p1, p0, q1, q0)
    v1 = p1 * q1
    v0 = p0 * q0
    v = (v1 + v0) / (q1 + q0)
    return np.sum(v1) / np.sum(v0) * np.sum(v * q0) / np.sum(v * q1)


@flatten_inputs
def agmean(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q0: npt.ArrayLike,
    elasticity: float,
    type: str = "arithmetic",
    skipnan: bool = False,
) -> np.float64:
    elasticity = float(elasticity)
    if not 0 <= elasticity <= 1:
        raise ValueError("`elasticity` must be between 0 and 1")
    if type not in ["arithmetic", "geometric"]:
        raise ValueError("`type` must be 'arithmetic' or 'geometric'")
    r = 1.0 if type == "arithmetic" else 0.0
    v0 = p0 * q0
    return nested_mean(
        p1 / p0, (v0, v0), order=(r, (0, 1)), outer_weights=(elasticity, 1 - elasticity)
    )
