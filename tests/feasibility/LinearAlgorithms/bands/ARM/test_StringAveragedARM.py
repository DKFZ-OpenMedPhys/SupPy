import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import StringAveragedARM
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


def test_StringAveragedARM_constructor_full(get_ARM_variables_full):
    """
    Test the constructor of the StringAveragedARM class with full
    matrices.
    """
    A, lb, ub = get_ARM_variables_full
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]])  # sequential like
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert alg.strings == [[0, 1]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])  # simultaneous like
    assert alg.strings == [[0], [1]]
    assert np.array_equal(alg.weights, np.ones(2) / 2)


def test_StringAveragedARM_constructor_sparse(get_ARM_variables_sparse):
    """
    Test the constructor of the StringAveragedARM class with sparse
    matrices.
    """
    A, lb, ub = get_ARM_variables_sparse
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]])  # sequential like
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert alg.strings == [[0, 1]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])  # simultaneous like
    assert alg.strings == [[0], [1]]
    assert np.array_equal(alg.weights, np.ones(2) / 2)


def test_StringAveragedARM_step_full_sequential_like(get_ARM_variables_full):
    """
    Test the step method of the StringAveragedARM class with full matrices
    for sequential like strings.
    """
    A, lb, ub = get_ARM_variables_full
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]])  # sequential like

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


def test_StringAveragedARM_step_sparse_sequential_like(get_ARM_variables_sparse):
    """
    Test the step method of the StringAveragedARM class with sparse matrices
    for sequential like strings.
    """
    A, lb, ub = get_ARM_variables_sparse
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]])  # sequential like

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


def test_StringAveragedARM_step_full_simultaneous_like(get_ARM_variables_sparse):
    """
    Test the step method of the StringAveragedARM class with full matrices
    for simultaneous like strings.
    """
    A, lb, ub = get_ARM_variables_sparse

    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])  # simultaneous like

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


def test_StringAveragedARM_step_sparse_simultaneous_like(get_ARM_variables_sparse):
    """
    Test the step method of the StringAveragedARM class with sparse matrices
    for simultaneous like strings.
    """
    A, lb, ub = get_ARM_variables_sparse

    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])  # simultaneous like

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
