import numpy as np
import numpy.typing as npt


Vector = npt.NDArray[np.float64]

def mean(
    x: Vector,
    weights: Vector | None = None,
    order: float = 1.0
) -> np.float64:
    return np.float64()

def nested_mean(
    x: Vector,
    weights: tuple[Vector, Vector] | None = None,
    order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0))
) -> np.float64:
    return np.float64()

def update_weights(
    x: Vector,
    weights: Vector,
    order: float = 1.0
) -> Vector:
    return np.array(1.0)

def transmute_weights(
    x: Vector,
    weights: Vector | None = None,
    from_order: float = 1.0,
    to_order: float = 1.0
) -> Vector:
    return np.array(1.0)

def nested_transmute_weights(
    x: Vector,
    weights: tuple[Vector, Vector] | None = None,
    from_order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    to_order: float = 1.0
) -> Vector:
    return np.array(1.0)

def nested_transmute_weights2(
    x: Vector,
    weights: tuple[Vector, Vector] | None = None,
    from_order: tuple[float, tuple[float, float]] = (0.0, (1.0, -1.0)),
    to_order: float = 1.0
) -> Vector:
    return np.array(1.0)
