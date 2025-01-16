import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialAMSHyperslab
from suppy.utils import LinearMapping


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
def get_SequentialAMSHyperslab_input_full(get_full_variables):
    A, lb, ub = get_full_variables
    return SequentialAMSHyperslab(A, lb, ub), A, lb, ub


@pytest.fixture
def get_SequentialAMSHyperslab_input_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return SequentialAMSHyperslab(A, lb, ub), A, lb, ub


def test_SequentialAMSHyperslab_no_relaxation_constructor_full(
    get_SequentialAMSHyperslab_input_full,
):
    """
    Test the SequentialAMSHyperslab constructor with no relaxation and a
    full
    matrix.
    """
    alg, A, lb, ub = get_SequentialAMSHyperslab_input_full

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_SequentialAMSHyperslab_no_relaxation_constructor_sparse(
    get_SequentialAMSHyperslab_input_sparse,
):
    """
    Test the SequentialAMSHyperslab constructor with no relaxation and a
    sparse
    matrix.
    """
    alg, A, lb, ub = get_SequentialAMSHyperslab_input_sparse

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.Bounds.l, lb)
    assert np.array_equal(alg.Bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_SequentialAMSHyperslab_constructor_wrong_bounds_shape(get_full_variables):
    """Test the SequentialAMSHyperslab constructor with wrong bounds shape."""
    A, lb, _ = get_full_variables
    ub = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        SequentialAMSHyperslab(A, lb, ub)

    lb = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        SequentialAMSHyperslab(A, lb, ub)


def test_SequentialAMSHyperslab_map_full(get_SequentialAMSHyperslab_input_full):
    """Test the map function of the SequentialAMSHyperslab class with full
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, 0, 1, 1]))


def test_SequentialAMSHyperslab_map_sparse(get_SequentialAMSHyperslab_input_sparse):
    """Test the map function of the SequentialAMSHyperslab class with sparse
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_sparse

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, 0, 1, 1]))


def test_SequentialAMSHyperslab_indexed_map_full(get_SequentialAMSHyperslab_input_full):
    """
    Test the indexed_map function of the SequentialAMSHyperslab class with
    full
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_SequentialAMSHyperslab_indexed_map_sparse(get_SequentialAMSHyperslab_input_sparse):
    """
    Test the indexed_map function of the SequentialAMSHyperslab class with
    sparse
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_sparse
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_SequentialAMSHyperslab_step_full(get_SequentialAMSHyperslab_input_full):
    """Test the step function of the SequentialAMSHyperslab class with full
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(np.abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(np.abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(np.abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(np.abs(x_n - np.array([-1, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(np.abs(x_n - np.array([-1.5, 0]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_SequentialAMSHyperslab_step_sparse(get_SequentialAMSHyperslab_input_sparse):
    """Test the step function of the SequentialAMSHyperslab class with
    sparse.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_sparse

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(np.abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(np.abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(np.abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(np.abs(x_n - np.array([-1, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(np.abs(x_n - np.array([-1.5, 0]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_SequentialAMSHyperslab_step_full_algoritimic_relaxation(get_full_variables):
    """Test the step function of the SequentialAMSHyperslab class with
    relaxation.
    """
    A, lb, ub = get_full_variables
    # test with relaxation
    alg = SequentialAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([0.5, 0.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_SequentialAMSHyperslab_step_sparse_algoritimic_relaxation(get_sparse_variables):
    """Test the step function of the SequentialAMSHyperslab class with
    relaxation.
    """
    A, lb, ub = get_sparse_variables
    # test with relaxation
    alg = SequentialAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([0.5, 0.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_SequentialAMSHyperslab_step_full_relaxation(get_full_variables):
    """Test the step function of the SequentialAMSHyperslab class with
    relaxation.
    """
    A, lb, ub = get_full_variables
    # test with relaxation
    alg = SequentialAMSHyperslab(A, lb, ub, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([0.5, 0.5])) < 1e-10)


def test_SequentialAMSHyperslab_step_sparse_relaxation(get_sparse_variables):
    """Test the step function of the SequentialAMSHyperslab class with
    relaxation.
    """
    A, lb, ub = get_sparse_variables
    # test with relaxation
    alg = SequentialAMSHyperslab(A, lb, ub, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([0.5, 0.5])) < 1e-10)


def test_SequentialAMSHyperslab_custom_cs(get_SequentialAMSHyperslab_input_full):
    """Test the step function of the SequentialAMSHyperslab class with
    custom.
    """
    _, A, lb, ub = get_SequentialAMSHyperslab_input_full

    # test with custom cs
    alg3 = SequentialAMSHyperslab(A, lb, ub, cs=[3, 2, 1, 0])

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg3.step(x_1)
    assert np.all(np.abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg3.step(x_2)
    assert np.all(np.abs(x_n - np.array([3 / 4, 5 / 4]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg3.step(x_3)
    assert np.all(np.abs(x_n - np.array([0, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg3.step(x_4)
    assert np.all(np.abs(x_n - np.array([-1, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg3.step(x_5)
    assert np.all(np.abs(x_n - np.array([-1.5, 0]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_SequentialAMSHyperslab_infinity_bounds(get_SequentialAMSHyperslab_input_full):
    """
    Test the step function of the SequentialAMSHyperslab class with infinity
    in
    bounds.
    """
    _, A, lb, _ = get_SequentialAMSHyperslab_input_full

    # test with infinity in bounds
    ub2 = np.array([np.inf, 2, 3 / 2, 3 / 2])
    alg4 = SequentialAMSHyperslab(A, lb, ub2)

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg4.step(x_1)
    assert np.all(np.abs(x_n - np.array([1.5, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg4.step(x_2)
    assert np.all(np.abs(x_n - np.array([1, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg4.step(x_3)
    assert np.all(np.abs(x_n - np.array([0.5, 1.5]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg4.step(x_4)
    assert np.all(np.abs(x_n - np.array([-1, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg4.step(x_5)
    assert np.all(np.abs(x_n - np.array([-1.5, 0]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_SequentialAMSHyperslab_proximity(get_SequentialAMSHyperslab_input_full):
    """Test the proximity function of the SequentialAMSHyperslab class."""
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    assert np.abs(alg.proximity(x_1) - 0.04) < 1e-10
    assert np.abs(alg.proximity(x_2) - 9 / 8) < 1e-10
    assert np.abs(alg.proximity(x_3) - 0.04) < 1e-10
    assert np.abs(alg.proximity(x_4) - 9 / 8) < 1e-10
    assert np.abs(alg.proximity(x_5) - 9 / 8) < 1e-10
