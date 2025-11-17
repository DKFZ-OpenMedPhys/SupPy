import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import ExtrapolatedLandweberHalfspace, AdaptiveStepLandweberHalfspace


@pytest.fixture
def get_full_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    ub = -1 * np.array([-2, -2, -3 / 2, -3 / 2])
    return np.concatenate((A, -A)), np.concatenate((ub, ub))


@pytest.fixture
def get_sparse_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    A = sparse.csr_matrix(np.concatenate((A, -A)))
    ub = -1 * np.array([-2, -2, -3 / 2, -3 / 2])
    return A, np.concatenate((ub, ub))


@pytest.fixture
def get_ExtrapolatedLandweberHalfspace_input_full(get_full_variables):
    A, ub = get_full_variables
    return ExtrapolatedLandweberHalfspace(A, ub), A, ub


@pytest.fixture
def get_ExtrapolatedLandweberHalfspace_input_sparse(get_sparse_variables):
    A, ub = get_sparse_variables
    return ExtrapolatedLandweberHalfspace(A, ub), A, ub


@pytest.fixture
def get_AdaptiveStepLandweberHalfspace_input_full(get_full_variables):
    A, ub = get_full_variables
    return AdaptiveStepLandweberHalfspace(A, ub), A, ub


@pytest.fixture
def get_AdaptiveStepLandweberHalfspace_input_sparse(get_sparse_variables):
    A, ub = get_sparse_variables
    return AdaptiveStepLandweberHalfspace(A, ub), A, ub


def test_ExtrapolatedLandweberHalfspace_project(get_ExtrapolatedLandweberHalfspace_input_full):

    alg, A, ub = get_ExtrapolatedLandweberHalfspace_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_ExtrapolatedLandweberHalfspace_project_sparse(
    get_ExtrapolatedLandweberHalfspace_input_sparse,
):

    alg, A, ub = get_ExtrapolatedLandweberHalfspace_input_sparse
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_AdaptiveStepLandweberHalfspace_project(get_AdaptiveStepLandweberHalfspace_input_full):

    alg, A, ub = get_AdaptiveStepLandweberHalfspace_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    print(proj)
    np.testing.assert_allclose(proj, np.array([19 / 12, 19 / 12]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))
