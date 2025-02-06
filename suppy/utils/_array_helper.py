"""General class for easier matrix operations."""
import numpy as np
from scipy import sparse

try:
    import cupy as cp
    import cupyx.scipy.sparse as csparse

    NO_GPU = False

except ImportError:
    NO_GPU = True
    cp = None
    csparse = None


class LinearMapping:
    """This class is used to allow interoperatibility between numpy, scipy etc."""

    def __init__(self, A):
        self.gpu = False  # flag for gpu

        if NO_GPU is False:  # is only checked when cupy is available

            if isinstance(A, cp.ndarray):
                self.flag = "cupy_full"
                self.gpu = True  # set a flag for gpu
                if A.ndim == 1:
                    self.A = cp.array([A])  # wrap 1d arrays
                elif A.ndim > 2:
                    raise ValueError("A must be a 2D array or a sparse matrix.")
                else:
                    self.A = A

            elif csparse.issparse(A):
                self.flag = "cupy_sparse"
                self.gpu = True  # set a flag for gpu
                self.A = csparse.csr_matrix(A)  # transform to csr format

        # set a flag based on class

        if isinstance(A, np.ndarray):
            self.flag = "numpy"
            if A.ndim == 1:
                self.A = np.array([A])  # wrap 1d arrays

            elif A.ndim > 2:
                raise ValueError("A must be a 2D array or a sparse matrix.")

            else:
                self.A = A

        elif sparse.issparse(A):
            self.flag = "scipy_sparse"
            self.A = sparse.csr_array(A)  # transform to csr format

    @staticmethod
    def get_flags(A):
        _gpu = False
        _flag = None
        if NO_GPU is False:  # is only checked when cupy is available

            if isinstance(A, cp.ndarray):
                _flag = "cupy_full"
                _gpu = True  # set a flag for gpu

            elif csparse.issparse(A):
                _flag = "cupy_sparse"
                _gpu = True  # set a flag for gpu

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

    def __hasattr__(self, attr):
        return hasattr(self.A, attr)

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
            return csparse.linalg.norm(self.A, ord=order) ** power
        raise ValueError("Unknown flag.")

    # def normalize_rows(self, order=None, power=1):
    #     """Normalize the rows of the matrix with the "norm" norm of power."""
    #     if self.flag == "numpy":
    #         return self.A / (np.linalg.norm(self.A, axis=1, ord=order) ** power)[:, None]

    #     elif self.flag == "scipy_sparse":
    #         return (
    #             sparse.diags_array(1 / (sparse.linalg.norm(self.A, axis=1, ord=order) ** power))
    #             @ self.A
    #         )

    #     elif self.flag == "cupy_full":
    #         return self.A / (cp.linalg.norm(self.A, axis=1, order=order) ** power)[:, None]

    #     elif self.flag == "cupy_sparse":
    #         return (
    #             csparse.diags(
    #                 (1 / (csparse.linalg.norm(self.A, axis=1, ord=order) ** power)).ravel()
    #             )
    #             @ self.A
    #         )

    def row_norm(self, order=None, power=1):
        """Get the row norms of the matrix."""
        if self.flag == "numpy":
            return np.linalg.norm(self.A, axis=1, ord=order) ** power

        if self.flag == "scipy_sparse":
            return sparse.linalg.norm(self.A, axis=1, ord=order) ** power

        if self.flag == "cupy_full":
            return cp.linalg.norm(self.A, axis=1, ord=order) ** power

        if self.flag == "cupy_sparse":
            return csparse.linalg.norm(self.A, axis=1, ord=order) ** power

        raise ValueError("Unknown flag.")

    def single_map(self, x, i):
        """Apply a linear map to a single row of the matrix."""
        if self.flag in ["numpy", "cupy_full"]:
            return self.A[i] @ x

        if self.flag in ["scipy_sparse", "cupy_sparse"]:
            idx1, idx2 = self.A.indptr[i], self.A.indptr[i + 1]
            return self.A.data[idx1:idx2] @ x[self.A.indices[idx1:idx2]]
        raise ValueError("Unknown flag.")

    def index_map(self, x, idx):
        """Apply a linear map to a subset of the matrix."""
        if self.flag in ["numpy", "cupy_full"]:
            return self.A[idx] @ x

        if self.flag in ["scipy_sparse", "cupy_sparse"]:
            return self.A[idx] @ x

        raise ValueError("Unknown flag.")

    def update_step(self, x, c, i):

        if self.flag in ["numpy", "cupy_full"]:
            x += self.A[i] * c

        elif self.flag in ["scipy_sparse", "cupy_sparse"]:
            idx1, idx2 = self.A.indptr[i], self.A.indptr[i + 1]
            x[self.A.indices[idx1:idx2]] += self.A.data[idx1:idx2] * c
        else:
            raise ValueError("Unknown flag.")

    def getrow(self, i):
        """Get a row of the matrix."""
        if self.flag in ["numpy", "cupy_full"]:
            return self.A[i]

        if self.flag in ["scipy_sparse", "cupy_sparse"]:
            return self.A.getrow(i)

        raise ValueError("Unknown flag.")
