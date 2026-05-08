import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import DROPHyperplane


@pytest.fixture
def get_sparse_variables():
    A = np.array([[1, 0], [1 / 2, -1 / 2], [0, 1]])
    A = sparse.csr_matrix(A)
    b = np.array([1, 0, 1])
    return A, b


@pytest.fixture
def get_DROPHyperplane_input_sparse(get_sparse_variables):
    A, b = get_sparse_variables
    return DROPHyperplane(A, b), A, b


def test_DROPHyperplane_project_sparse(get_DROPHyperplane_input_sparse):
    alg, A, b = get_DROPHyperplane_input_sparse
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.5, 1.5]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.5, 0.5]))
