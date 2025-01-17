import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SimultaneousARM
from suppy.utils import LinearMapping


@pytest.fixture
def get_ARM_variables_full():
    A = np.array([[1, 1], [-1, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    return A, lb, ub


@pytest.fixture
def get_ARM_variables_sparse():
    A = sparse.csr_matrix([[1, 1], [-1, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    return A, lb, ub


def test_SimultaneousARM_constructor_full(get_ARM_variables_full):
    """Test the SimultaneousARM constructor."""
    A, lb, ub = get_ARM_variables_full

    # check without given weights
    alg = SimultaneousARM(A, lb, ub)
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.weights, np.ones(len(A)) / len(A))

    alg = SimultaneousARM(A, lb, ub, weights=np.array([1 / 3, 2 / 3]))
    assert np.array_equal(alg.weights, np.array([1 / 3, 2 / 3]))
    # check with given weights
    alg = SimultaneousARM(A, lb, ub, weights=np.ones(len(A)))
    assert np.array_equal(alg.weights, np.ones(len(A)) / len(A))

    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SimultaneousARM_constructor_sparse(get_ARM_variables_sparse):
    """Test the SimultaneousARM constructor."""
    A, lb, ub = get_ARM_variables_sparse

    # check without given weights
    alg = SimultaneousARM(A, lb, ub)
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])

    alg = SimultaneousARM(A, lb, ub, weights=np.array([1 / 3, 2 / 3]))
    assert np.array_equal(alg.weights, np.array([1 / 3, 2 / 3]))
    # check with given weights
    alg = SimultaneousARM(A, lb, ub, weights=np.ones(A.shape[0]))
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])

    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SimultaneousARM_step_full(get_ARM_variables_full):
    """Test the step function of the SimultaneousARM class."""
    A, lb, ub = get_ARM_variables_full
    alg = SimultaneousARM(A, lb, ub)

    x_1 = np.array([0.0, 0.0])
    x_2 = np.array([1.0, 1.0])
    x_3 = np.array([-1.0, -1.0])
    x_4 = np.array([-1.0, 1.0])
    x_5 = np.array([0.0, 2.0])

    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([0.0, 0.0])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg.step(x_2)
    assert np.all(np.abs(x_n - 13 / 16 * np.array([1.0, 1.0])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(np.abs(x_n - 13 / 16 * np.array([-1.0, -1.0])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(np.abs(x_n - 13 / 16 * np.array([-1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(np.abs(x_n - np.array([0, 13 / 8]))) < 1e-10


def test_SimultaneousARM_step_sparse(get_ARM_variables_sparse):
    """Test the step function of the SimultaneousARM class."""
    A, lb, ub = get_ARM_variables_sparse

    alg = SimultaneousARM(A, lb, ub, weights=np.ones(A.shape[0]))

    x_1 = np.array([0.0, 0.0])
    x_2 = np.array([1.0, 1.0])
    x_3 = np.array([-1.0, -1.0])
    x_4 = np.array([-1.0, 1.0])
    x_5 = np.array([0.0, 2.0])

    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([0.0, 0.0])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg.step(x_2)
    assert np.all(np.abs(x_n - 13 / 16 * np.array([1.0, 1.0])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(np.abs(x_n - 13 / 16 * np.array([-1.0, -1.0])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(np.abs(x_n - 13 / 16 * np.array([-1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(np.abs(x_n - np.array([0, 13 / 8])) < 1e-10)
