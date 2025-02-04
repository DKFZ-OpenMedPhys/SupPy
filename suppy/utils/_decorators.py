"""File that includes somme decorators used throughout the module."""

import warnings
from typing import Callable
import numpy as np


def ensure_float_array(func: Callable) -> Callable:
    """
    Decorator to ensure that the input array is of type float32 or float64.
    If the input array is not of type float32 or float64, it will be converted
    to float64.

    Parameters
    ----------
    func : Callable
        The function to be decorated.

    Returns
    -------
    Callable
        The decorated function which ensures the input array is of type float32 or float64.

    Raises
    ------
    TypeError
        If the input array cannot be converted to float64.

    Warnings
    --------
    UserWarning
        If the input array is not of type float32 or float64 and needs to be converted.
    """

    def wrapper(self, arr, *args, **kwargs):
        if arr.dtype not in [np.float32, np.float64]:
            warnings.warn("Array is not of type float32 or float64, converting to float64")
            try:
                arr = arr.astype(np.float64)
            except Exception as e:
                raise TypeError("Failed to convert array to float64") from e
        return func(self, arr, *args, **kwargs)

    return wrapper
