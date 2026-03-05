from typing import Callable
import numpy as np

try:
    import cupy as cp

    NO_GPU = False
except ImportError:
    cp = np
    NO_GPU = True


class FuncWrapper:
    """
    A callable class for a function that keeps track of the number of times
    it is called.

    Parameters
    ----------
    func : Callable
        The function to be wrapped.
    args : list
        The arguments to be passed to the function.

    Attributes
    ----------
    func : Callable
        The function to be wrapped.
    args : list
        The arguments to be passed to the function.
    fcount : int
        The number of times the function has been called.
    """

    def __init__(self, func: Callable, args=[]):
        self.func = func
        self.args = args
        self.fcount = 0
        self._intermediate_x = None
        self._intermediate_value = 0.0

    def __call__(self, x):
        xp = cp if isinstance(x, cp.ndarray) else np
        self.fcount += 1
        if self._intermediate_x is not None and xp.array_equal(x, self._intermediate_x):
            return self._intermediate_value
        else:
            self._intermediate_x = x.copy()
            self._intermediate_value = self.func(x, *self.args)
            return self._intermediate_value
