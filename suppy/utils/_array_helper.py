"""General class for easier matrix operations."""
import numpy as np
from scipy import sparse
import warnings

try:
    import cupy as cp
    import cupyx.scipy.sparse as csparse
    import cupyx.scipy.sparse.linalg as clinalg

    NO_GPU = False

except ImportError:
    NO_GPU = True
    cp = None
    csparse = None


def to_host_floats(*values) -> list[float]:
    """
    Convert scalars to Python floats with at most one device-to-host
    transfer.

    Every transfer of a GPU value to the host forces a device
    synchronization, so all GPU values are collected and transferred
    together.

    Parameters
    ----------
    *values : float or array-like
        Python, NumPy or CuPy scalars (or arrays with a single element).

    Returns
    -------
    list[float]
        The values as Python floats, in the given order.
    """
    out = list(values)
    if not NO_GPU:
        on_gpu = [i for i, v in enumerate(values) if isinstance(v, cp.ndarray)]
        if on_gpu:
            fetched = cp.stack([values[i].reshape(()).astype(cp.float64) for i in on_gpu]).get()
            for j, i in enumerate(on_gpu):
                out[i] = fetched[j]
    return [float(v) for v in out]


def to_host_list(values: list) -> list:
    """
    Host copies of a list of scalars or arrays with at most one
    device-to-host transfer.

    Parameters
    ----------
    values : list
        Python, NumPy or CuPy scalars or arrays (of floating point type).

    Returns
    -------
    list
        The values with every CuPy array replaced by its NumPy counterpart.
    """
    out = list(values)
    if not NO_GPU:
        on_gpu = [i for i, v in enumerate(values) if isinstance(v, cp.ndarray)]
        if on_gpu:
            flat = cp.concatenate([values[i].ravel().astype(cp.float64) for i in on_gpu]).get()
            start = 0
            for i in on_gpu:
                size = values[i].size
                out[i] = flat[start : start + size].reshape(values[i].shape)
                start += size
    return out


def stack_to_numpy(values: list) -> np.ndarray:
    """
    Stack a list of scalars or equally shaped arrays into one NumPy array.

    GPU values are stacked on the device first, so that only one
    device-to-host transfer is needed.

    Parameters
    ----------
    values : list
        Python, NumPy or CuPy scalars or arrays.

    Returns
    -------
    np.ndarray
        The stacked values.
    """
    if not NO_GPU and any(isinstance(v, cp.ndarray) for v in values):
        return cp.stack([cp.asarray(v) for v in values]).get()
    return np.array(values)


class LinearMapping:
    """This class is used to allow interoperatibility between numpy, scipy etc."""

    def __init__(self, A):
        self.gpu = False

        if not NO_GPU:
            if isinstance(A, cp.ndarray):
                self.flag = "cupy_full"
                self.gpu = True
                if A.ndim == 1:
                    self.A = cp.array([A])
                elif A.ndim > 2:
                    raise ValueError("A must be a 2D array or a sparse matrix.")
                else:
                    self.A = A
                return

            elif csparse.issparse(A):
                self.flag = "cupy_sparse"
                self.gpu = True
                if isinstance(A, csparse.csr_matrix):
                    self.A = A
                else:
                    warnings.warn(
                        "Converting to csr matrix format. This creates a copy of your matrix!",
                        stacklevel=2,
                    )
                    self.A = csparse.csr_matrix(A)
                # host copy of the row pointers: row-wise access (sequential methods)
                # would otherwise need a device synchronization for every row
                self._indptr_host = self.A.indptr.get()
                return

        if isinstance(A, np.ndarray):
            self.flag = "numpy"
            if A.ndim == 1:
                self.A = np.array([A])
            elif A.ndim > 2:
                raise ValueError("A must be a 2D array or a sparse matrix.")
            else:
                self.A = A

        elif sparse.issparse(A):
            self.flag = "scipy_sparse"
            if isinstance(A, sparse.csr_array):
                self.A = A
            else:
                warnings.warn(
                    "Converting to csr array format. This creates a copy of your array!",
                    stacklevel=2,
                )
                self.A = sparse.csr_array(A)

        else:
            raise ValueError(
                f"Unsupported matrix type: {type(A)}. Expected np.ndarray, scipy sparse, "
                "or (if CuPy is available) cp.ndarray or cupy sparse."
            )

    @staticmethod
    def get_flags(A):
        _gpu = False
        _flag = None
        if not NO_GPU:
            if isinstance(A, cp.ndarray):
                return "cupy_full", True
            elif csparse.issparse(A):
                return "cupy_sparse", True

        if isinstance(A, np.ndarray):
            _flag = "numpy"
        elif sparse.issparse(A):
            _flag = "scipy_sparse"
        return _flag, _gpu

    # Representation
    def __str__(self):
        return self.A.__str__()

    def __repr__(self):
        return self.A.__repr__()

    # Attribute access

    def __getattr__(self, attr):
        return getattr(self.A, attr)

    # Get and set elements
    def __getitem__(self, key):
        return self.A[key]

    def __setitem__(self, key, value):
        self.A[key] = value

    def __eq__(self, other):
        return self.A == other

    # Mathematical operators

    def __add__(self, other):
        return self.A + other

    def __radd__(self, other):
        return other + self.A

    def __sub__(self, other):
        return self.A - other

    def __rsub__(self, other):
        return other - self.A

    def __mul__(self, other):
        return self.A * other

    def __rmul__(self, other):
        return other * self.A

    def __truediv__(self, other):
        return self.A / other

    def __rtruediv__(self, other):
        return other / self.A

    def __pow__(self, other):
        return self.A**other

    def __matmul__(self, other):
        return self.A @ other

    def __rmatmul__(self, other):
        return other @ self.A

    def __iter__(self):
        return self.A.__iter__()

    def __len__(self):
        return self.A.__len__()

    def get_norm(self, order=None, power=1):
        """Get the norm of the matrix."""
        if self.flag == "numpy":
            return np.linalg.norm(self.A, ord=order) ** power

        if self.flag == "scipy_sparse":
            return sparse.linalg.norm(self.A, ord=order) ** power

        if self.flag == "cupy_full":
            return cp.linalg.norm(self.A, ord=order) ** power

        if self.flag == "cupy_sparse":
            return clinalg.norm(self.A, ord=order) ** power
        raise ValueError("Unknown flag.")

    def _sparse_axis_norm(self, axis, order, power):
        """
        Vector norm along one axis of a sparse matrix.

        Equivalent to ``scipy.sparse.linalg.norm(A, axis=axis, ord=order) **
        power`` and its cupy counterpart, but shares the index arrays instead
        of copying the matrix, allowing a reduction of peak memory usage.
        Orders other than 1, 2 and ``None`` fall back to the library routine.

        Parameters
        ----------
        axis : int
            0 for column norms, 1 for row norms.
        order : int or None
            Order of the vector norm.
        power : int or float
            Exponent applied to the norm.

        Returns
        -------
        npt.NDArray
            The norms along *axis*, raised to *power*.
        """
        xp = cp if self.flag == "cupy_sparse" else np
        linalg = clinalg if self.flag == "cupy_sparse" else sparse.linalg

        if order in (None, 2):
            data = self.A.data * self.A.data
        elif order == 1:
            data = xp.abs(self.A.data)
        else:
            return linalg.norm(self.A, axis=axis, ord=order) ** power

        # Same indices and indptr, so no copy of the pattern is made.
        transformed = type(self.A)((data, self.A.indices, self.A.indptr), shape=self.A.shape)
        summed = xp.asarray(transformed.sum(axis=axis)).ravel()

        if order == 1:
            return summed**power
        # The root is redundant for power=2 but keeps the result bit-identical
        # to the library, which some callers are tested against exactly.
        return xp.sqrt(summed) ** power

    def row_norm(self, order=None, power=1):
        """Get the row norms of the matrix."""
        if self.flag == "numpy":
            return np.linalg.norm(self.A, axis=1, ord=order) ** power

        if self.flag in ("scipy_sparse", "cupy_sparse"):
            return self._sparse_axis_norm(1, order, power)

        if self.flag == "cupy_full":
            return cp.linalg.norm(self.A, axis=1, ord=order) ** power

        raise ValueError("Unknown flag.")

    def column_norm(self, order=None, power=1):
        """Get the column norms of the matrix."""
        if self.flag == "numpy":
            return np.linalg.norm(self.A, axis=0, ord=order) ** power

        if self.flag in ("scipy_sparse", "cupy_sparse"):
            return self._sparse_axis_norm(0, order, power)

        if self.flag == "cupy_full":
            return cp.linalg.norm(self.A, axis=0, ord=order) ** power

        raise ValueError("Unknown flag.")

    def single_map(self, x, i):
        """Apply a linear map to a single row of the matrix."""
        if self.flag in ["numpy", "cupy_full"]:
            return self.A[i] @ x

        if self.flag in ["scipy_sparse", "cupy_sparse"]:
            idx1, idx2 = self._row_bounds(i)
            return self.A.data[idx1:idx2] @ x[self.A.indices[idx1:idx2]]
        raise ValueError("Unknown flag.")

    def index_map(self, x, idx):
        """Apply a linear map to a subset of the matrix."""
        return self.A[idx] @ x

    def update_step(self, x, c, i):
        """Add a scaled row of the matrix to x in-place: x += A[i] * c."""
        if self.flag in ["numpy", "cupy_full"]:
            x += self.A[i] * c

        elif self.flag in ["scipy_sparse", "cupy_sparse"]:
            idx1, idx2 = self._row_bounds(i)
            x[self.A.indices[idx1:idx2]] += self.A.data[idx1:idx2] * c
        else:
            raise ValueError("Unknown flag.")

    def _row_bounds(self, i):
        """Start and end of row `i` in the data of a sparse matrix (host ints)."""
        indptr = self._indptr_host if self.flag == "cupy_sparse" else self.A.indptr
        i = int(i)
        return int(indptr[i]), int(indptr[i + 1])

    def getrow(self, i):
        """Get a row of the matrix."""
        if self.flag in ["numpy", "cupy_full"]:
            return self.A[i]

        if self.flag in ["scipy_sparse", "cupy_sparse"]:
            return self.A.getrow(i)

        raise ValueError("Unknown flag.")
