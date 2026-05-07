## helper functions for radiotherapy

import numpy as np
import numpy.typing as npt
import scipy

from Testing._ignore.aaaaa import grad

try:
    import cupy as cp
    from cupyx.scipy.sparse import isspmatrix

    NO_GPU = False

except ImportError:
    NO_GPU = True
    cp = np


class SquaredDeviation:
    """Mean squared deviation of dose from a reference dose level.

    Penalizes both over- and under-dosing symmetrically:

        f(d) = (1/N) * sum_{i in idxs} (d_i - d_ref)^2

    Parameters
    ----------
    d_ref : float
        Reference dose value.
    idxs : list or npt.ArrayLike
        Voxel indices (boolean mask or integer array) over which the
        objective is evaluated.
    """

    def __init__(self, d_ref: float, idxs: list):
        self.d_ref = d_ref
        self.idxs = idxs
        self.length = self.idxs.sum() if self.idxs.dtype == bool else len(self.idxs)

    def objective_value(self, x: npt.ArrayLike) -> float:
        """Return the mean squared deviation at dose vector *x*."""
        diff = x[self.idxs] - self.d_ref
        return 1 / self.length * (diff @ diff)

    def objective_gradient(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Return the gradient of the objective with respect to *x*."""
        xp = cp if isinstance(x, cp.ndarray) else np
        grad = xp.zeros(x.shape[0])
        grad[self.idxs] = 2 * (x[self.idxs] - self.d_ref)
        return 1 / self.length * grad


class SquaredOverdosing:
    """Mean squared overdosing penalty.

    Only penalizes voxels that receive more dose than the reference:

        f(d) = (1/N) * sum_{i in idxs} max(d_i - d_ref, 0)^2

    Parameters
    ----------
    d_ref : float
        Reference dose value.
    idxs : list or npt.ArrayLike
        Voxel indices (boolean mask or integer array) over which the
        objective is evaluated.
    """

    def __init__(self, d_ref: float, idxs: list):
        self.d_ref = d_ref
        self.idxs = idxs
        self.length = self.idxs.sum() if self.idxs.dtype == bool else len(self.idxs)

    def objective_value(self, x: npt.ArrayLike) -> float:
        """Return the mean squared overdosing at dose vector *x*."""
        d_diff = x[self.idxs] - self.d_ref
        d_diff[d_diff < 0] = 0
        return 1 / self.length * (d_diff @ d_diff)

    def objective_gradient(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Return the gradient of the objective with respect to *x*."""
        xp = cp if isinstance(x, cp.ndarray) else np
        grad = xp.zeros(x.shape[0])
        d_diff = x[self.idxs] - self.d_ref
        d_diff[d_diff < 0] = 0
        grad[self.idxs] = 2 * d_diff
        return 1 / self.length * grad


class EUD:
    """Generalized Equivalent Uniform Dose (gEUD).

    Computes the power-mean dose over the selected voxels:

        f(d) = (1/N * sum_{i in idxs} d_i^a)^(1/a)

    For a > 1 the metric is sensitive to hot spots (OAR use); for a < 1 it
    is sensitive to cold spots (target use).

    Parameters
    ----------
    exponent : float
        Power-law exponent *a*.
    idxs : list or npt.ArrayLike
        Voxel indices (boolean mask or integer array) over which the
        objective is evaluated.
    """

    def __init__(self, exponent: float, idxs: list):
        self.exponent = exponent
        self.idxs = idxs
        self.length = self.idxs.sum() if self.idxs.dtype == bool else len(self.idxs)

    def objective_value(self, x: npt.ArrayLike) -> float:
        """Return the gEUD at dose vector *x*."""
        return (1 / self.length * ((x[self.idxs] ** self.exponent).sum())) ** (1 / self.exponent)

    def objective_gradient(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Return the gradient of the objective with respect to *x*."""
        xp = cp if isinstance(x, cp.ndarray) else np
        grad = xp.zeros(x.shape[0])

        if x @ x == 0:
            return grad
        grad[self.idxs] = (
            ((x[self.idxs] ** self.exponent).sum()) ** (1 / self.exponent - 1)
            * (x[self.idxs] ** (self.exponent - 1))
            / self.length ** (1 / self.exponent)
        )
        return grad


class MeanDose:
    """Mean dose over a set of voxels.

        f(d) = (1/N) * sum_{i in idxs} d_i

    Parameters
    ----------
    idxs : list or npt.ArrayLike
        Voxel indices (boolean mask or integer array) over which the
        objective is evaluated.
    """

    def __init__(self, idxs: list):
        self.idxs = idxs
        self.length = self.idxs.sum() if self.idxs.dtype == bool else len(self.idxs)

    def objective_value(self, x: npt.ArrayLike) -> float:
        """Return the mean dose at dose vector *x*."""
        return 1 / self.length * ((x[self.idxs]).sum())

    def objective_gradient(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Return the gradient of the objective with respect to *x*."""
        xp = cp if isinstance(x, cp.ndarray) else np
        grad = xp.zeros(x.shape[0])
        grad[self.idxs] = 1 / self.length * np.ones(self.length)
        return grad


class MaxDVH:
    """Dose-volume histogram (DVH) objective penalizing overdosing.

    Enforces that at most *max_v_percent* % of the voxel volume receives
    a dose exceeding *d_ref*. Voxels with dose in [d_ref, d_max] are
    penalized, where d_max is the (100 - max_v_percent)-th dose percentile:

        f(d) = (1/N) * sum_{i: d_ref <= d_i <= d_max} (d_i - d_ref)^2

    Parameters
    ----------
    d_ref : float
        Reference dose value; doses above this threshold are penalized.
    max_v_percent : float
        Maximum allowable percentage of volume receiving dose above *d_ref*.
    idxs : list or npt.ArrayLike
        Voxel indices (boolean mask or integer array) over which the
        objective is evaluated.
    """

    def __init__(self, d_ref: float, max_v_percent: float, idxs: list):
        self.d_ref = d_ref
        self.max_v_percent = max_v_percent
        self.idxs = idxs
        self.length = self.idxs.sum() if self.idxs.dtype == bool else len(self.idxs)

    def objective_value(self, x: npt.ArrayLike) -> float:
        """Return the MaxDVH objective value at dose vector *x*."""
        xp = cp if isinstance(x, cp.ndarray) else np
        d = x[self.idxs]
        d_diff = d - self.d_ref

        # calculate the D_max_percent value
        d_max = xp.percentile(
            d, 100 - self.max_v_percent
        )  # , method = 'interpolated_inverted_cdf')
        d_diff[(d < self.d_ref) | (d > d_max)] = 0
        return 1 / self.length * (d_diff @ d_diff)

    def objective_gradient(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Return the gradient of the objective with respect to *x*."""
        xp = cp if isinstance(x, cp.ndarray) else np
        grad = xp.zeros(x.shape[0])
        d = x[self.idxs]
        d_diff = d - self.d_ref

        d_max = xp.percentile(
            d, 100 - self.max_v_percent
        )  # , method = 'interpolated_inverted_cdf')
        d_diff[(d < self.d_ref) | (d > d_max)] = 0
        grad[self.idxs] = (2 / self.length) * d_diff

        return grad


class MinDVH:
    """Dose-volume histogram (DVH) objective penalizing underdosing.

    Enforces that at least *min_v_percent* % of the voxel volume receives
    a dose of at least *d_ref*. Voxels with dose in [d_min, d_ref] are
    penalized, where d_min is the *min_v_percent*-th dose percentile:

        f(d) = (1/N) * sum_{i: d_min <= d_i <= d_ref} (d_i - d_ref)^2

    Parameters
    ----------
    d_ref : float
        Reference dose value; doses below this threshold are penalized.
    min_v_percent : float
        Minimum required percentage of volume receiving dose at least *d_ref*.
    idxs : list or npt.ArrayLike
        Voxel indices (boolean mask or integer array) over which the
        objective is evaluated.
    """

    def __init__(self, d_ref: float, min_v_percent: float, idxs: list):
        self.d_ref = d_ref
        self.min_v_percent = min_v_percent
        self.idxs = idxs
        self.length = self.idxs.sum() if self.idxs.dtype == bool else len(self.idxs)

    def objective_value(self, x: npt.ArrayLike) -> float:
        """Return the MinDVH objective value at dose vector *x*."""
        xp = cp if isinstance(x, cp.ndarray) else np
        d = x[self.idxs]
        d_diff = d - self.d_ref

        # calculate the D_min_percent value
        d_min = xp.percentile(d, self.min_v_percent)  # , method = 'interpolated_inverted_cdf')
        d_diff[(d > self.d_ref) | (d < d_min)] = 0
        return 1 / self.length * (d_diff @ d_diff)

    def objective_gradient(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Return the gradient of the objective with respect to *x*."""
        xp = cp if isinstance(x, cp.ndarray) else np
        grad = xp.zeros(x.shape[0])
        d = x[self.idxs]
        d_diff = d - self.d_ref

        d_min = xp.percentile(d, self.min_v_percent)  # , method = 'interpolated_inverted_cdf')
        d_diff[(d > self.d_ref) | (d < d_min)] = 0
        grad[self.idxs] = (2 / self.length) * d_diff

        return grad


class objectives:
    """Weighted sum of radiotherapy dose objectives.

    Combines multiple individual objective functions with penalty weights and
    applies the dose-influence matrix *dij* to map fluence *x* to dose *d*:

        F(x) = sum_k penalty_k * f_k(dij @ x)

    Parameters
    ----------
    objectives : list
        List of objective instances (e.g. :class:`SquaredDeviation`,
        :class:`EUD`), each exposing ``objective_value`` and
        ``objective_gradient`` methods.
    penalties : list
        Scalar penalty weights, one per objective. Must have the same
        length as *objectives*.
    dij : npt.ArrayLike
        Dose-influence matrix mapping fluence vector to dose vector.
    store_csc_copy : bool, optional
        If ``True`` and a GPU sparse matrix is detected, stores a CSC copy
        of *dij* to accelerate the gradient back-projection. By default
        ``False``.
    """

    def __init__(
        self, objectives: list, penalties: list, dij: npt.ArrayLike, store_csc_copy: bool = False
    ):
        self.objectives = objectives
        self.penalties = penalties
        if len(self.objectives) != len(self.penalties):
            raise ValueError(
                f"Number of objectives is {len(self.objectives)}, but number of penalties is {len(self.penalties)}."
            )
        self.dij = dij
        self.store_csc_copy = store_csc_copy and not NO_GPU and isspmatrix(self.dij)
        if self.store_csc_copy:
            self.dij_csc = cp.sparse.csc_matrix(dij).copy()

    def objective_value(self, x: npt.ArrayLike) -> float:
        """Return the weighted sum of objective values at fluence vector
        *x*.
        """
        d = self.dij @ x
        return sum([f.objective_value(d) * p for f, p in zip(self.objectives, self.penalties)])

    def objective_gradient(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """Return the gradient of the weighted sum with respect to *x*."""
        d = self.dij @ x

        if self.store_csc_copy:
            return (
                sum([f.objective_gradient(d) * p for f, p in zip(self.objectives, self.penalties)])
                @ self.dij_csc
            )

        return (
            sum([f.objective_gradient(d) * p for f, p in zip(self.objectives, self.penalties)])
            @ self.dij
        )
