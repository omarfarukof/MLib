import numpy as np
import pandas as pd


def to_numpy(data: pd.DataFrame| np.ndarray| list[float]| float| int , dtype=float) -> np.ndarray:
        if isinstance(data, np.ndarray):
            return data.astype(dtype)
        elif isinstance(data, pd.DataFrame):
            return data.values
        elif isinstance(data, list):
            return np.array(data , dtype=float)
        elif isinstance(data, float) or isinstance(data, int):
            return np.array([data] , dtype=float)
        elif data is None:
            return np.array([])
        else:
            raise TypeError("data is not in proper format")


