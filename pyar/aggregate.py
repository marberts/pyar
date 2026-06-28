import pandas as pd
import numpy as np


class AggregationStructure:
    def __init__(self) -> None:
        return None

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> None:
        return None

    @staticmethod
    def from_list(list: list) -> None:
        return None

    @property
    def weights(self):
        return None


class PriceIndex:
    def __init__(self) -> None:
        return None

    @staticmethod
    def from_pandas(df: pd.DataFrame) -> None:
        return None

    @staticmethod
    def from_numpy(array: np.ndarray) -> None:
        return None


class ChainablePriceIndex(PriceIndex):
    def __init__(self) -> None:
        return None
