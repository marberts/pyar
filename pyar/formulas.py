import numpy as np
import numpy.typing as npt
from means import mean, nested_mean


def carli(p1: npt.ArrayLike, p0: npt.ArrayLike, drop_na=False) -> np.ndarray:
    return mean(p1 / p0, drop_na=drop_na)

def jevons(p1: npt.ArrayLike, p0: npt.ArrayLike, drop_na=False) -> np.ndarray:
    return mean(p1 / p0, order=0, drop_na=drop_na)

def coggshall(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    drop_na=False
) -> np.ndarray:
    return mean(p1 / p0, order=-1, drop_na=drop_na)

def fisher(
    p1: npt.ArrayLike,
    p0: npt.ArrayLike,
    q1: npt.ArrayLike,
    q0: npt.ArrayLike,
    drop_na=False
) -> np.float64:
    return nested_mean(p1 / p0, p0 * q0, p1 * q1, drop_na=drop_na)
