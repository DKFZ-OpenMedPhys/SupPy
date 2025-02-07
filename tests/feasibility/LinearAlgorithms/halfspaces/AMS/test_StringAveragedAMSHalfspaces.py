import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import StringAveragedAMSHalfspace
from suppy.utils import LinearMapping


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


def test_StringAveragedAMSHalfspace_constructor_full(get_full_variables):
    """Test the StringAveragedAMSHalfspace constructor."""
    A, b = get_full_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])  # sequential like

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert alg.strings == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0], [1], [2], [3], [4], [5], [6], [7]]
    )  # simultaneous like
    assert alg.strings == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert np.array_equal(alg.weights, np.ones(8) / 8)


def test_StringAveragedAMSHalfspace_constructor_sparse(get_sparse_variables):
    """Test the StringAveragedAMSHalfspace constructor."""
    A, b = get_sparse_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])  # sequential like

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert alg.strings == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0], [1], [2], [3], [4], [5], [6], [7]]
    )  # simultaneous like
    assert alg.strings == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert np.array_equal(alg.weights, np.ones(8) / 8)


def test_StringAveragedAMSHalfspace_step_full_sequential_like(get_full_variables):
    """
    Test the step function of the StringAveragedAMSHalfspace class for
    sequential
    like strings.
    """

    A, b = get_full_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])  # sequential like

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_StringAveragedAMSHalfspace_step_sparse_sequential_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHalfspace class for
    sequential
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])  # sequential like

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_StringAveragedAMSHalfspace_step_full_simultaneous_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHalfspace class for
    simultaneous
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0], [1], [2], [3], [4], [5], [6], [7]]
    )  # simultaneous like

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([29 / 16, 29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.175, -1.175])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_StringAveragedAMSHalfspace_step_sparse_simultaneous_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHalfspace class for
    simultaneous
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0], [1], [2], [3], [4], [5], [6], [7]]
    )  # simultaneous like

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([29 / 16, 29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.175, -1.175])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_5)
