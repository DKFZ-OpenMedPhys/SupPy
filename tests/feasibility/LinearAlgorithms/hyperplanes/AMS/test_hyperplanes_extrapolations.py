import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import AdaptiveStepLandweberHyperplane, ExtrapolatedLandweberHyperplane


@pytest.fixture
def get_full_variables():
    A = np.array([[1, 0], [1 / 2, -1 / 2], [0, 1]])
    b = np.array([1, 0, 1])
    return A, b


@pytest.fixture
def get_sparse_variables():
    A = np.array([[1, 0], [1 / 2, -1 / 2], [0, 1]])
    A = sparse.csr_matrix(A)
    b = np.array([1, 0, 1])
    return A, b


@pytest.fixture
def get_ExtrapolatedLandweberHyperplane_input_full(get_full_variables):
    A, ub = get_full_variables
    return ExtrapolatedLandweberHyperplane(A, ub), A, ub


@pytest.fixture
def get_ExtrapolatedLandweberHyperplane_input_sparse(get_sparse_variables):
    A, ub = get_sparse_variables
    return ExtrapolatedLandweberHyperplane(A, ub), A, ub


@pytest.fixture
def get_AdaptiveStepLandweberHyperplane_input_full(get_full_variables):
    A, ub = get_full_variables
    return AdaptiveStepLandweberHyperplane(A, ub), A, ub


@pytest.fixture
def get_AdaptiveStepLandweberHyperplane_input_sparse(get_sparse_variables):
    A, ub = get_sparse_variables
    return AdaptiveStepLandweberHyperplane(A, ub), A, ub


def test_ExtrapolatedLandweberHyperplane_project(get_ExtrapolatedLandweberHyperplane_input_full):
    alg, A, ub = get_ExtrapolatedLandweberHyperplane_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    print(proj)
    assert np.array_equal(proj, np.array([1.0, 1.0]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.0, 1.0]))


def test_AdaptiveLandweberHyperplane_project(get_AdaptiveStepLandweberHyperplane_input_full):
    alg, A, ub = get_AdaptiveStepLandweberHyperplane_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.0, 1.0]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.0, 1.0]))
