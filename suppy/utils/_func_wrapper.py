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
    cache : bool, optional
        If True, the last input and its function value are stored and the
        stored value is returned when the function is called again with an
        equal input, by default True. Checking the input requires a full
        comparison and a copy of it on every call, which on the GPU forces a
        device synchronization. Disable it if equal consecutive inputs are
        rare.

    Attributes
    ----------
    func : Callable
        The function to be wrapped.
    args : list
        The arguments to be passed to the function.
    cache : bool
        Whether the last function value is cached.
    fcount : int
        The number of calls of the wrapper. Without cache this is the number
        of function evaluations; with cache, calls answered from the cache
        are counted as well.
    """

    def __init__(self, func: Callable, args=None, cache: bool = True):
        self.func = func
        self.args = args if args is not None else []
        self.cache = cache
        self.fcount = 0
        self._intermediate_x = None
        self._intermediate_value = 0.0

    def __call__(self, x):
        self.fcount += 1
        if not self.cache:
            return self.func(x, *self.args)

        xp = cp if isinstance(x, cp.ndarray) else np
        if (
            self._intermediate_x is not None
            and x is not None
            and xp.array_equal(x, self._intermediate_x)
        ):
            return self._intermediate_value
        else:
            if x is None:  # should mainly happen when evaluating perturbation scheme for
                self._intermediate_x = None
            else:
                self._intermediate_x = x.copy()
            self._intermediate_value = self.func(x, *self.args)
            return self._intermediate_value
