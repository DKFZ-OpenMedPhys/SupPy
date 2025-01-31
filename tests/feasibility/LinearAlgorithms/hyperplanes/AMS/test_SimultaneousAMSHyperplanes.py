import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SimultaneousAMSHyperplane
from suppy.utils import LinearMapping


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


def test_SimultaneousAMSHyperplane_no_relaxation_no_weights_constructor_full(get_full_variables):
    """
    Test the SimultaneousAMSHyperplane constructor with no relaxation and a
    full
    matrix.
    """
    A, b = get_full_variables
    alg = SimultaneousAMSHyperplane(A, b)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(len(A)) / len(A))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SimultaneousAMSHyperplane_no_relaxation_no_weights_constructor_sparse(
    get_sparse_variables,
):
    """
    Test the SimultaneousAMSHyperplane constructor with no relaxation and a
    sparse
    matrix.
    """
    A, b = get_sparse_variables
    alg = SimultaneousAMSHyperplane(A, b)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def testSimultaneousAMSHyperplane_relaxation_weights_constructor_full(get_full_variables):
    """
    Test the SimultaneousAMSHyperplane constructor with relaxation and
    custom
    weights.
    """
    A, b = get_full_variables
    alg = SimultaneousAMSHyperplane(
        A, b, algorithmic_relaxation=1.5, relaxation=1.5, weights=np.ones(len(A))
    )

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(len(A)) / A.shape[0])
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.5


def testSimultaneousAMSHyperplane_relaxation_weights_constructor_sparse(get_sparse_variables):
    """
    Test the SimultaneousAMSHyperplane constructor with relaxation and
    custom
    weights.
    """
    A, b = get_sparse_variables
    alg = SimultaneousAMSHyperplane(
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


def test_SimultaneousAMSHyperplane_step_full(get_full_variables):
    """Test the step function of the SimultaneousAMSHyperplane class."""
    A, b = get_full_variables
    alg = SimultaneousAMSHyperplane(A, b, weights=np.ones(len(A)) / A.shape[0])

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([5 / 3, 5 / 3])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([5 / 3, 5 / 3])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([4 / 3, 2])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([5 / 6, 11 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1 / 3, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-7 / 6, -1 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SimultaneousAMSHyperplane_step_sparse(get_sparse_variables):
    """Test the step function of the SimultaneousAMSHyperplane class."""
    A, b = get_sparse_variables

    alg = SimultaneousAMSHyperplane(A, b, weights=np.ones(A.shape[0]))

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([5 / 3, 5 / 3])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([5 / 3, 5 / 3])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([4 / 3, 2])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([5 / 6, 11 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1 / 3, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-7 / 6, -1 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SimultaneousAMSHyperplane_algorithmic_relaxation_step(get_sparse_variables):
    """Test the step function with relaxation of the SimultaneousAMSHyperplane
    class.
    """
    A, b = get_sparse_variables

    alg = SimultaneousAMSHyperplane(A, b, weights=np.ones(A.shape[0]), algorithmic_relaxation=2.0)

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([4 / 3, 4 / 3])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([4 / 3, 4 / 3])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([5 / 3, 1])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([10 / 6, 4 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([4 / 3, 0])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([4 / 6, -2 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SimultaneousAMSHyperplane_relaxation_step(get_sparse_variables):
    """Test the step function with relaxation of the SimultaneousAMSHyperplane
    class.
    """
    A, b = get_sparse_variables

    alg = SimultaneousAMSHyperplane(A, b, weights=np.ones(A.shape[0]), relaxation=2.0)
    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_n = alg.project(x_1)
    assert np.all(abs(x_n - np.array([4 / 3, 4 / 3])) < 1e-10)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([5 / 3, 1])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([10 / 6, 4 / 6])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([4 / 3, 0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([4 / 6, -2 / 6])) < 1e-10)


def test_SimultaneousAMSHyperplane_proximity(get_full_variables):
    """Test the proximity function of the SimultaneousAMSHyperplane class."""
    A, b = get_full_variables
    alg = SimultaneousAMSHyperplane(A, b, weights=np.ones(len(A)) / len(A))

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    assert abs(alg.proximity(x_1) - 2 / 3) < 1e-10
    assert abs(alg.proximity(x_2) - 5 / 3) < 1e-10
    assert abs(alg.proximity(x_3) - 7.25 / 3) < 1e-10
    assert abs(alg.proximity(x_4) - 14 / 3) < 1e-10
    assert abs(alg.proximity(x_5) - 19.25 / 3) < 1e-10
