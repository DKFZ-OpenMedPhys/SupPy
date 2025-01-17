import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialARM
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


def test_SequentialARM_constructor_full(get_ARM_variables_full):
    """Test the SequentialARM constructor."""
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0
    assert alg._k == 0


def test_SequentialARM_constructor_sparse(get_ARM_variables_sparse):
    """Test the SequentialARM constructor."""
    A, lb, ub = get_ARM_variables_sparse
    alg = SequentialARM(A, lb, ub)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0
    assert alg._k == 0


def test_SequentialARM_step_full(get_ARM_variables_full):
    """Test the step function of the SequentialARM class."""
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub)

    x_1 = np.array([0.0, 0.0])
    x_2 = np.array([1.0, 1.0])
    x_3 = np.array([-1.0, -1.0])
    x_4 = np.array([-1.0, 1.0])
    x_5 = np.array([0.0, 2.0])

    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([0.0, 0.0])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg.step(x_2)
    assert np.all(np.abs(x_n - 5 / 8 * np.array([1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(np.abs(x_n - 5 / 8 * np.array([-1.0, -1.0])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(np.abs(x_n - 5 / 8 * np.array([-1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(np.abs(x_n - np.array([0, 5 / 4])) < 1e-10)


def test_SequentialARM_step_sparse(get_ARM_variables_sparse):
    """Test the step function of the SequentialARM class."""
    A, lb, ub = get_ARM_variables_sparse
    alg = SequentialARM(A, lb, ub)

    x_1 = np.array([0.0, 0.0])
    x_2 = np.array([1.0, 1.0])
    x_3 = np.array([-1.0, -1.0])
    x_4 = np.array([-1.0, 1.0])
    x_5 = np.array([0.0, 2.0])

    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([0.0, 0.0])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg.step(x_2)
    assert np.all(np.abs(x_n - 5 / 8 * np.array([1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(np.abs(x_n - 5 / 8 * np.array([-1.0, -1.0])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(np.abs(x_n - 5 / 8 * np.array([-1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(np.abs(x_n - np.array([0, 5 / 4])) < 1e-10)
