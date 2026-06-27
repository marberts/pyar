import numpy as np
from means import mean, scale_weights


def hallerbach(p1, p0, v1, v0):
    v0, v1 = scale_weights(v0), scale_weights(v1)
    laspeyres, paasche = mean(p1 / p0, v0), mean(p1 / p0, v1, order=-1)
    (
        0.5 * np.sqrt(paasche / laspeyres) * v0
        + 0.5 * np.sqrt(laspeyres / paasche) * v1 / p0
    )


def vonBortkiewicz(p1, p0, v1, v0):
    q1, q0 = v1 / p1, v0 / p0
    price, quantity = mean(p1 / p0, v0), mean(q1 / q0, v1)
    return scale_weights(v0) * (p1 / p0 / price - 1) * (q1 / q0 / quantity - 1)


def geo_vonBortkiewicz(p1, p0, v1, v0):
    v1, v0 = scale_weights(v1), scale_weights(v0)
    return v0 * (v1 / v0 - 1) * np.log(p1 / p0 / mean(p1 / p0, v0, order=0))
