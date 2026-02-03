import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import ExtrapolatedLandweberHyperslab, AdaptiveStepLandweberHyperslab


@pytest.fixture
def get_full_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    return A, lb, ub


@pytest.fixture
def get_sparse_variables():
    A = sparse.csr_matrix([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    return A, lb, ub


@pytest.fixture
def get_ExtrapolatedLandweberHyperslab_input_full(get_full_variables):
    A, lb, ub = get_full_variables
    return ExtrapolatedLandweberHyperslab(A, lb, ub), A, lb, ub


@pytest.fixture
def get_ExtrapolatedLandweberHyperslab_input_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return ExtrapolatedLandweberHyperslab(A, lb, ub), A, lb, ub


@pytest.fixture
def get_AdaptiveStepLandweberHyperslab_input_full(get_full_variables):
    A, lb, ub = get_full_variables
    return AdaptiveStepLandweberHyperslab(A, lb, ub), A, lb, ub


@pytest.fixture
def get_AdaptiveStepLandweberHyperslab_input_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return AdaptiveStepLandweberHyperslab(A, lb, ub), A, lb, ub


def test_ExtrapolatedLandweberHyperslab_project(get_ExtrapolatedLandweberHyperslab_input_full):
    alg, A, lb, ub = get_ExtrapolatedLandweberHyperslab_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_ExtrapolatedLandweberHyperslab_project_sparse(
    get_ExtrapolatedLandweberHyperslab_input_sparse,
):
    alg, A, lb, ub = get_ExtrapolatedLandweberHyperslab_input_sparse
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_AdaptiveStepLandweberHyperslab_project(get_AdaptiveStepLandweberHyperslab_input_full):
    alg, A, lb, ub = get_AdaptiveStepLandweberHyperslab_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_AdaptiveStepLandweberHyperslab_project_sparse(
    get_AdaptiveStepLandweberHyperslab_input_sparse,
):
    alg, A, lb, ub = get_AdaptiveStepLandweberHyperslab_input_sparse
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))
