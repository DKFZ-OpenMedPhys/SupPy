import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SimultaneousAMSHalfspace
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


def test_SimultaneousAMSHalfspace_no_relaxation_no_weights_constructor_full(get_full_variables):
    """
    Test the SimultaneousAMSHalfspace constructor with no relaxation and a
    full
    matrix.
    """
    A, b = get_full_variables
    alg = SimultaneousAMSHalfspace(A, b)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(len(A)) / len(A))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SimultaneousAMSHalfspace_no_relaxation_no_weights_constructor_sparse(
    get_sparse_variables,
):
    """
    Test the SimultaneousAMSHalfspace constructor with no relaxation and a
    sparse
    matrix.
    """
    A, b = get_sparse_variables
    alg = SimultaneousAMSHalfspace(A, b)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def testSimultaneousAMSHalfspace_relaxation_weights_constructor_full(get_full_variables):
    """
    Test the SimultaneousAMSHalfspace constructor with relaxation and custom
    weights.
    """
    A, b = get_full_variables
    alg = SimultaneousAMSHalfspace(
        A, b, algorithmic_relaxation=1.5, relaxation=1.5, weights=np.ones(len(A))
    )

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(len(A)) / A.shape[0])
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.5


def testSimultaneousAMSHalfspace_relaxation_weights_constructor_sparse(get_sparse_variables):
    """
    Test the SimultaneousAMSHalfspace constructor with relaxation and custom
    weights.
    """
    A, b = get_sparse_variables
    alg = SimultaneousAMSHalfspace(
        A,
        b,
        algorithmic_relaxation=1.5,
        relaxation=1.5,
        weights=np.ones(A.shape[0]),
    )

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.5


def test_SimultaneousAMSHalfspace_step_full(get_full_variables):
    """Test the step function of the SimultaneousAMSHalfspace class."""
    A, b = get_full_variables
    alg = SimultaneousAMSHalfspace(A, b, weights=np.ones(len(A)) / A.shape[0])

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


def test_SimultaneousAMSHalfspace_step_sparse(get_sparse_variables):
    """Test the step function of the SimultaneousAMSHalfspace class."""
    A, b = get_sparse_variables

    alg = SimultaneousAMSHalfspace(A, b, weights=np.ones(A.shape[0]))

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


def test_SimultaneousAMSHalfspace_algorithmic_relaxation_step(get_sparse_variables):
    """Test the step function with relaxation of the SimultaneousAMSHalfspace
    class.
    """
    A, b = get_sparse_variables

    alg = SimultaneousAMSHalfspace(A, b, weights=np.ones(A.shape[0]), algorithmic_relaxation=2.0)

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.15, 1.15])) < 1e-10)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.15, 1.15])) < 1e-10)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([1.625, 1.625])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.15, -1.15])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n + np.array([1.625, 1.625])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([1.625, -1.625])) < 1e-10)


def test_SimultaneousAMSHalfspace_relaxation_step(get_sparse_variables):
    """Test the step function with relaxation of the SimultaneousAMSHalfspace
    class.
    """
    A, b = get_sparse_variables

    alg = SimultaneousAMSHalfspace(A, b, weights=np.ones(A.shape[0]), relaxation=2.0)

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.15, 1.15])) < 1e-10)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.15, 1.15])) < 1e-10)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([1.625, 1.625])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.15, -1.15])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n + np.array([1.625, 1.625])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([1.625, -1.625])) < 1e-10)


def test_SimultaneousAMSHalfspace_proximity(get_full_variables):
    """Test the proximity function of the SimultaneousAMSHalfspace class."""
    A, b = get_full_variables
    alg = SimultaneousAMSHalfspace(A, b, weights=np.ones(len(A)) / len(A))

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    prox_measures = []
    no_prox = alg.proximity(x_1, prox_measures)
    assert no_prox.size == 0

    prox_measures = [("p_norm", 2), "max_norm"]
    prox_1 = alg.proximity(x_1, prox_measures)
    prox_2 = alg.proximity(x_2, prox_measures)
    prox_3 = alg.proximity(x_3, prox_measures)
    prox_4 = alg.proximity(x_4, prox_measures)
    prox_5 = alg.proximity(x_5, prox_measures)

    assert np.abs(prox_1[0] - 0.02) < 1e-10
    assert np.abs(prox_1[1] - 0.4) < 1e-10

    assert np.abs(prox_2[0] - 9 / 16) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 0.02) < 1e-10
    assert np.abs(prox_3[1] - 0.4) < 1e-10

    assert np.abs(prox_4[0] - 9 / 16) < 1e-10
    assert np.abs(prox_4[1] - 2) < 1e-10

    assert np.abs(prox_5[0] - 9 / 16) < 1e-10
    assert np.abs(prox_5[1] - 2) < 1e-10


def test_SimultaneousAMSHalfspace_proximity_weighted(get_full_variables):
    A, b = get_full_variables
    alg = SimultaneousAMSHalfspace(
        A, b, weights=np.array([1 / 4, 1 / 8, 1 / 16, 1 / 16, 1 / 4, 1 / 8, 1 / 16, 1 / 16])
    )

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    prox_measures = []
    no_prox = alg.proximity(x_1, prox_measures)
    assert no_prox.size == 0

    prox_measures = [("p_norm", 2), "max_norm"]
    print(alg.A, alg.b, alg.weights)
    prox_1 = alg.proximity(x_1, prox_measures)
    prox_2 = alg.proximity(x_2, prox_measures)
    prox_3 = alg.proximity(x_3, prox_measures)
    prox_4 = alg.proximity(x_4, prox_measures)
    prox_5 = alg.proximity(x_5, prox_measures)

    assert np.abs(prox_1[0] - 0.04) < 1e-10
    assert np.abs(prox_1[1] - 0.4) < 1e-10
    print(prox_2)
    assert np.abs(prox_2[0] - 33 / 32) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 0.04) < 1e-10
    assert np.abs(prox_3[1] - 0.4) < 1e-10

    assert np.abs(prox_4[0] - 33 / 32) < 1e-10
    assert np.abs(prox_4[1] - 2) < 1e-10

    assert np.abs(prox_5[0] - 17 / 32) < 1e-10
    assert np.abs(prox_5[1] - 2) < 1e-10
