import numpy as np
import pandas as pd
from .means import (
    mean,
    nested_mean,
    _balance_nas,
    scale_weights,
    extended_mean,
)


def carli(p1: pd.Series, p0: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, order=1.0, skipna=skipna)


def dutot(p1: pd.Series, p0: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, p0, order=1.0, skipna=skipna)


def jevons(p1: pd.Series, p0: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, order=0.0, skipna=skipna)


def cswd(p1: pd.Series, p0: pd.Series, skipna=False) -> float:
    return nested_mean(p1 / p0, order=(0.0, (1.0, -1.0)), skipna=skipna)


def balk_walsh(p1: pd.Series, p0: pd.Series, skipna=False) -> float:
    return nested_mean(p1 / p0, order=(0.0, (0.5, 0.5)), skipna=skipna)


def hybrid_cswd(p1: pd.Series, p0: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, np.sqrt(p0 / p1), order=1.0, skipna=skipna)


def coggshall(p1: pd.Series, p0: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, order=-1.0, skipna=skipna)


def laspeyres(p1: pd.Series, p0: pd.Series, q0: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, p0 * q0, order=1.0, skipna=skipna)


def palgrave(p1: pd.Series, p0: pd.Series, q1: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, p1 * q1, order=1.0, skipna=skipna)


def paasche(p1: pd.Series, p0: pd.Series, q1: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, p1 * q1, order=-1.0, skipna=skipna)


def geo_laspeyres(p1: pd.Series, p0: pd.Series, q0: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, p0 * q0, order=0.0, skipna=skipna)


def geo_paasche(p1: pd.Series, p0: pd.Series, q1: pd.Series, skipna=False) -> float:
    return mean(p1 / p0, p1 * q1, order=0.0, skipna=skipna)


def fisher(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    return nested_mean(
        p1 / p0, (p0 * q0, p1 * q1), order=(0.0, (1.0, -1.0)), skipna=skipna
    )


def tornqvist(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    return nested_mean(
        p1 / p0, (p0 * q0, p1 * q1), order=(0.0, (0.0, 0.0)), skipna=skipna
    )


def drobisch(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    return nested_mean(
        p1 / p0, (p0 * q0, p1 * q1), order=(1.0, (1.0, -1.0)), skipna=skipna
    )


def walsh1(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    return mean(p1 / p0, p0 * np.sqrt(q0 * q1), order=1.0, skipna=skipna)


def marshall_edgeworth(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    return mean(p1 / p0, p0 * (q0 + q1), order=1.0, skipna=skipna)


def geary_khamis(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    return mean(p1 / p0, p0 * (1 / q0 + 1 / q1), order=1.0, skipna=skipna)


def walsh2(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    return mean(p1 / p0, np.sqrt(p1 * q1 * p0 * q0), order=0.0, skipna=skipna)


def theil(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    if skipna:
        p1, p0, q1, q0 = _balance_nas(p1, p0, q1, q0)
    w1 = scale_weights(p1 * q1)
    w0 = scale_weights(p0 * q0)
    return mean(p1 / p0, (w1 * w0 * (w1 + w0) / 2) ** (1 / 3), order=0.0)


def rao(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna=False,
) -> float:
    if skipna:
        p1, p0, q1, q0 = _balance_nas(p1, p0, q1, q0)
    w1 = scale_weights(p1 * q1)
    w0 = scale_weights(p0 * q0)
    return mean(p1 / p0, w1 * w0 / (w1 + w0) / 2, order=0.0)


def sato_vartia(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna: bool = False,
) -> float:
    return mean(
        p1 / p0,
        extended_mean(p0 * q0 / np.sum(p0 * q0), p1 * q1 / np.sum(p1 * q1)),
        order=0.0,
        skipna=skipna,
    )


def lloyd_moulton(
    p1: pd.Series,
    p0: pd.Series,
    q0: pd.Series,
    elasticity: float,
    skipna: bool = False,
) -> float:
    return mean(p1 / p0, p0 * q0, order=elasticity, skipna=skipna)


def lehr(
    p1: pd.Series,
    p0: pd.Series,
    q1: pd.Series,
    q0: pd.Series,
    skipna: bool = False,
) -> float:
    if skipna:
        p1, p0, q1, q0 = _balance_nans(p1, p0, q1, q0)
    v1 = p1 * q1
    v0 = p0 * q0
    v = (v1 + v0) / (q1 + q0)
    return np.sum(v1) / np.sum(v0) * np.sum(v * q0) / np.sum(v * q1)


def agmean(
    p1: pd.Series,
    p0: pd.Series,
    q0: pd.Series,
    elasticity: float,
    type: str = "arithmetic",
    skipna: bool = False,
) -> float:
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
