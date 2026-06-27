from means import scale_weights
import numpy as np
import numpy.typing as npt
from means import mean, nested_mean


def carli(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, order=1, skipnan=skipnan)


def dutot(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, p0, order=1, skipnan=skipnan)


def jevons(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, order=0, skipnan=skipnan)


def cswd(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return nested_mean(p1 / p0, order=(0, (1, -1)), skipnan=skipnan)


def hybrid_cswd(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, np.sqrt(p0 / p1), order=1, skipnan=skipnan)


def coggshall(p1: npt.ArrayLike, p0: npt.ArrayLike, skipnan=False) -> np.ndarray:
    return mean(p1 / p0, order=-1, skipnan=skipnan)


def laspeyres(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q0: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p0 * q0, order=1, skipnan=skipnan)


def palgrave(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q1: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p1 * q1, order=1, skipnan=skipnan)


def paasche(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q1: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p1 * q1, order=-1, skipnan=skipnan)


def geo_laspeyres(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q0: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p0 * q0, order=0, skipnan=skipnan)


def geo_paasche(
    p1: npt.ArrayLike, p0: npt.ArrayLike, q1: npt.ArrayLike, skipnan=False
) -> np.ndarray:
    return mean(p1 / p0, p1 * q1, order=0, skipnan=skipnan)


def fisher(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return nested_mean(p1 / p0, (p0 * q0, p1 * q1), order=(0, (1, -1)), skipnan=skipnan)


def tornqvist(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return nested_mean(p1 / p0, (p0 * q0, p1 * q1), order=(0, (0, 0)), skipnan=skipnan)


def drobisch(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return nested_mean(p1 / p0, (p0 * q0, p1 * q1), order=(1, (1, -1)), skipnan=skipnan)


def walsh1(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return mean(p1 / p0, p0 * np.sqrt(q0 * q1), order=1, skipnan=skipnan)


def marshall_edgeworth(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return mean(p1 / p0, p0 * (q0 + q1), order=1, skipnan=skipnan)


def geary_khamis(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return mean(p1 / p0, p0 * (1 / q0 + 1 / q1), order=1, skipnan=skipnan)


def walsh2(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    return mean(p1 / p0, np.sqrt(p1 * q1 * p0 * q0), order=0, skipnan=skipnan)


def theil(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    w1 = scale_weights(p1 * q1)
    w0 = scale_weights(p0 * q0)
    return mean(p1 / p0, (w1 * w0 * (w1 + w0) / 2) ** (1 / 3), order=0, skipnan=skipnan)


def rao(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    skipnan=False,
) -> np.float64:
    w1 = scale_weights(p1 * q1)
    w0 = scale_weights(p0 * q0)
    return mean(p1 / p0, w1 * w0 / (w1 + w0) / 2, order=0, skipnan=skipnan)
