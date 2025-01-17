import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialART3plus
from suppy.utils import LinearMapping


@pytest.fixture
def get_ART3_variables_full():
    A = np.array([[1, 0], [0, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    return A, lb, ub


@pytest.fixture
def get_ART3_variables_sparse():
    A = sparse.csr_matrix([[1, 0], [0, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    return A, lb, ub


def test_SequentialART3plus_constructor_full(get_ART3_variables_full):
    """Test the SequentialART3plus constructor."""
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub, cs=[1, 0])
    assert alg.cs == [1, 0]

    alg = SequentialART3plus(A, lb, ub)
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg._feasible == True


def test_SequentialART3plus_constructor_sparse(get_ART3_variables_sparse):
    """Test the SequentialART3plus constructor."""
    A, lb, ub = get_ART3_variables_sparse
    alg = SequentialART3plus(A, lb, ub)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg._feasible == True


def test_SequentialART3plus_step_full(get_ART3_variables_full):
    """Test the step function of the SequentialART3plus class."""
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub)

    x_1 = np.array([-3.0, 0.0])
    x_2 = np.array([-1.5, 0.0])
    x_3 = np.array([0.5, 0.5])
    x_4 = np.array([1.5, 0.0])
    x_5 = np.array([3.0, 0.0])
    x_6 = np.array([-3.0, -3.0])
    x_7 = np.array([-3.0, 3.0])
    x_8 = np.array([-3, 1.5])

    x_n = alg.step(x_1)
    assert np.array_equal(x_1, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_2)
    assert np.array_equal(x_2, x_n)
    assert np.array_equal(x_n, np.array([-0.5, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_3)
    assert np.array_equal(x_3, x_n)
    assert np.array_equal(x_n, np.array([0.5, 0.5]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_4)
    assert np.array_equal(x_4, x_n)
    assert np.array_equal(x_n, np.array([0.5, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_5)
    assert np.array_equal(x_5, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_6)
    assert np.array_equal(x_6, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_7)
    assert np.array_equal(x_7, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_8)
    assert np.array_equal(x_8, x_n)
    assert np.array_equal(x_n, np.array([0, 0.5]))


def test_SequentialART3plus_step_sparse(get_ART3_variables_sparse):
    """Test the step function of the SequentialART3plus class."""
    A, lb, ub = get_ART3_variables_sparse
    alg = SequentialART3plus(A, lb, ub)

    x_1 = np.array([-3.0, 0.0])
    x_2 = np.array([-1.5, 0.0])
    x_3 = np.array([0.5, 0.5])
    x_4 = np.array([1.5, 0.0])
    x_5 = np.array([3.0, 0.0])
    x_6 = np.array([-3.0, -3.0])
    x_7 = np.array([-3.0, 3.0])
    x_8 = np.array([-3, 1.5])

    x_n = alg.step(x_1)
    assert np.array_equal(x_1, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_2)
    assert np.array_equal(x_2, x_n)
    assert np.array_equal(x_n, np.array([-0.5, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_3)
    assert np.array_equal(x_3, x_n)
    assert np.array_equal(x_n, np.array([0.5, 0.5]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_4)
    assert np.array_equal(x_4, x_n)
    assert np.array_equal(x_n, np.array([0.5, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_5)
    assert np.array_equal(x_5, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_6)
    assert np.array_equal(x_6, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_7)
    assert np.array_equal(x_7, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_8)
    assert np.array_equal(x_8, x_n)
    assert np.array_equal(x_n, np.array([0, 0.5]))


if __name__ == "__main__":
    A = np.array([[1, 0], [0, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    test_SequentialART3plus_step_full((A, lb, ub))
