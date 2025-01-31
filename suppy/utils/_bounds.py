import numpy as np
import numpy.typing as npt
from typing import List

try:
    import cupy as cp

    no_gpu = False
except ImportError:
    no_gpu = True
    cp = np


class Bounds:
    """
    A class to help with hyperslab calculations.

    Parameters
    ----------
    lb : None or array_like, optional
        Lower bounds. If None, defaults to negative infinity if `ub` is provided.
    ub : None or array_like, optional
        Upper bounds. If None, defaults to positive infinity if `lb` is provided.

    Attributes
    ----------
    l : array_like
        Lower bounds.
    u : array_like
        Upper bounds.
    half_distance : array_like
        Half the distance between lower and upper bounds.
    center : array_like
        Center point between lower and upper bounds.

    Raises
    ------
    ValueError
        If the sizes of the lower and upper bounds do not match.
        If any lower bound is greater than the corresponding upper bound.
    """

    def __init__(self, lb: None | npt.ArrayLike = None, ub: None | npt.ArrayLike = None):
        # TODO: Rework validity check? Should be possible to just pass a scaler
        # TODO: default values for lower and upper bounds and check
        if lb is None and ub is not None:
            lb = -np.inf
        elif ub is None and lb is not None:
            ub = np.inf

        elif lb is None and ub is None:
            raise ValueError("At least one of the bounds must be provided")

        self.l = lb
        self.u = ub
        self.half_distance = self._half_distance()
        self.center = self._center()

    def residual(self, x: npt.ArrayLike):
        """
        Calculate the residuals between the input vector `x` and the bounds
        `l` and `u`.

        Parameters
        ----------
        x : npt.ArrayLike
            Input vector for which the residuals are to be calculated.

        Returns
        -------
        tuple of npt.ArrayLike
            A tuple containing two arrays:
            - The residuals between `x` and the lower bound `l`.
            - The residuals between the upper bound `u` and `x`.
        """
        return x - self.l, self.u - x

    def single_residual(self, x: float, i: int):
        """
        Calculate the residuals for a given value for a specific constraint
        with respect to the lower and upper bounds.

        Parameters
        ----------
        x : float
            The value for which the residuals are calculated.
        i : int
            The index of the bounds to use.

        Returns
        -------
        tuple of float
            A tuple containing the residuals (x - lower_bound, upper_bound - x).
        """
        return x - self.l[i], self.u[i] - x

    def indexed_residual(self, x: npt.ArrayLike, i: List[int] | npt.ArrayLike):
        """
        Compute the residuals for the given indices.

        Parameters
        ----------
        x : npt.ArrayLike
            The input array.
        i : List[int] or npt.ArrayLike
            The indices for which to compute the residuals.

        Returns
        -------
        tuple of npt.ArrayLike
            A tuple containing two arrays:
            - The residuals of `x` with respect to the lower bounds.
            - The residuals of `x` with respect to the upper bounds.
        """
        return x - self.l[i], self.u[i] - x

    def _center(self):
        """
        Calculate the center point between the lower bound (self.l) and the
        upper bound (self.u).

        Returns
        -------
        float
            The midpoint value between self.l and self.u.
        """
        return (self.l + self.u) / 2

    def _half_distance(self):
        """
        Calculate half the distance between the upper and lower bounds.

        Returns
        -------
        float
            Half the distance between the upper bound (self.u) and the lower bound (self.l).
        """
        return (self.u - self.l) / 2

    def project(self, x: npt.ArrayLike):
        """
        Project the input array `x` onto the bounds defined by `self.l` and
        `self.u`.

        Parameters
        ----------
        x : npt.ArrayLike
            Input array to be projected.

        Returns
        -------
        npt.ArrayLike
            The projected array where each element is clipped to be within the bounds
            defined by `self.l` and `self.u`.
        """
        xp = cp if isinstance(x, cp.ndarray) else np
        return xp.minimum(self.u, xp.maximum(self.l, x))
