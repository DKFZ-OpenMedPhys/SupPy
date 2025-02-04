import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import BlockIterativeAMSHyperslab
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
def get_BlockIterativeAMSHyperslab_input_full_sequential(get_full_variables):
    A, lb, ub = get_full_variables
    return BlockIterativeAMSHyperslab(A, lb, ub, weights=np.eye(4)), A, lb, ub


@pytest.fixture
def get_BlockIterativeAMSHyperslab_input_full_simultaneous(get_full_variables):
    A, lb, ub = get_full_variables
    return (
        BlockIterativeAMSHyperslab(A, lb, ub, weights=np.array([[1 / 4, 1 / 4, 1 / 4, 1 / 4]])),
        A,
        lb,
        ub,
    )


@pytest.fixture
def get_BlockIterativeAMSHyperslab_input_sparse_sequential(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return BlockIterativeAMSHyperslab(A, lb, ub, weights=np.eye(4)), A, lb, ub


@pytest.fixture
def get_BlockIterativeAMSHyperslab_input_sparse_simultaneous(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return (
        BlockIterativeAMSHyperslab(A, lb, ub, weights=np.array([[1 / 4, 1 / 4, 1 / 4, 1 / 4]])),
        A,
        lb,
        ub,
    )


def test_BlockIterativeAMSHyperslab_sequential_constructor_full(
    get_BlockIterativeAMSHyperslab_input_full_sequential,
):
    """
    Test the BlockIterativeAMSHyperslab constructor with sequential weights
    and a
    full matrix.
    """
    alg, A, lb, ub = get_BlockIterativeAMSHyperslab_input_full_sequential

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.weights, np.array([[1], [1], [1], [1]]))
    assert np.array_equal(alg.block_idxs, np.array([[0], [1], [2], [3]]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeAMSHyperslab_simultaneous_constructor_full(
    get_BlockIterativeAMSHyperslab_input_full_simultaneous,
):
    """
    Test the BlockIterativeAMSHyperslab constructor with simultaneous
    weights and a
    full matrix.
    """
    alg, A, lb, ub = get_BlockIterativeAMSHyperslab_input_full_simultaneous

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.weights, np.array([[1 / 4, 1 / 4, 1 / 4, 1 / 4]]))
    assert np.array_equal(alg.block_idxs, np.array([[0, 1, 2, 3]]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeAMSHyperslab_sequential_constructor_sparse(
    get_BlockIterativeAMSHyperslab_input_sparse_sequential,
):
    """
    Test the BlockIterativeAMSHyperslab constructor with sequential weights
    and a
    sparse matrix.
    """
    alg, A, lb, ub = get_BlockIterativeAMSHyperslab_input_sparse_sequential

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.weights, np.array([[1], [1], [1], [1]]))
    assert np.array_equal(alg.block_idxs, np.array([[0], [1], [2], [3]]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeAMSHyperslab_simultaneous_constructor_sparse(
    get_BlockIterativeAMSHyperslab_input_sparse_simultaneous,
):
    """
    Test the BlockIterativeAMSHyperslab constructor with simultaneous
    weights and a
    sparse matrix.
    """
    alg, A, lb, ub = get_BlockIterativeAMSHyperslab_input_sparse_simultaneous

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.weights, np.array([[1 / 4, 1 / 4, 1 / 4, 1 / 4]]))
    assert np.array_equal(alg.block_idxs, np.array([[0, 1, 2, 3]]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeAMSHyperslab_map_full(get_BlockIterativeAMSHyperslab_input_full_sequential):
    """
    Test the map function of the BlockIterativeAMSHyperslab class with
    sequential
    weights and full matrix.
    """
    alg, _, _, _ = get_BlockIterativeAMSHyperslab_input_full_sequential

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, 0, 1, 1]))


def test_BlockIterativeAMSHyperslab_map_sparse(
    get_BlockIterativeAMSHyperslab_input_sparse_sequential,
):
    """
    Test the map function of the BlockIterativeAMSHyperslab class with
    sequential
    weights and sparse matrix.
    """
    alg, _, _, _ = get_BlockIterativeAMSHyperslab_input_sparse_sequential

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, 0, 1, 1]))


def test_BlockIterativeAMSHyperslab_indexed_map_full(
    get_BlockIterativeAMSHyperslab_input_full_sequential,
):
    """
    Test the indexed_map function of the BlockIterativeAMSHyperslab class
    with
    sequential weights and full matrix.
    """
    alg, _, _, _ = get_BlockIterativeAMSHyperslab_input_full_sequential
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_BlockIterativeAMSHyperslab_indexed_map_sparse(
    get_BlockIterativeAMSHyperslab_input_sparse_sequential,
):
    """
    Test the indexed_map function of the BlockIterativeAMSHyperslab class
    with
    sequential weights and sparse matrix.
    """
    alg, _, _, _ = get_BlockIterativeAMSHyperslab_input_sparse_sequential
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_BlockIterativeAMSHyperslab_sequential_step_full(
    get_BlockIterativeAMSHyperslab_input_full_sequential,
):
    """
    Test the step function of the BlockIterativeAMSHyperslab class with
    sequential
    weights and full matrix.
    """
    alg, _, _, _ = get_BlockIterativeAMSHyperslab_input_full_sequential

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


def test_BlockIterativeAMSHyperslab_simultaneous_step_full(
    get_BlockIterativeAMSHyperslab_input_full_simultaneous,
):
    """
    Test the step function of the BlockIterativeAMSHyperslab class with
    simultaneous
    weights and full matrix.
    """
    alg, _, _, _ = get_BlockIterativeAMSHyperslab_input_full_simultaneous

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.15, 1.15])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.15, 1.15])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - 1 / 4 * np.array([6.5, 6.5])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.15, -1.15])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - 1 / 4 * np.array([-6.5, -6.5])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - 1 / 4 * np.array([6.5, -6.5])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_BlockIterativeAMSHyperslab_simultaneous_step_sparse(
    get_BlockIterativeAMSHyperslab_input_sparse_simultaneous,
):
    """
    Test the step function of the BlockIterativeAMSHyperslab class with
    simultaneous
    weights and sparse matrix.
    """
    alg, _, _, _ = get_BlockIterativeAMSHyperslab_input_sparse_simultaneous

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.15, 1.15])) < 1e-10)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - 1 / 4 * np.array([6.5, 6.5])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.15, -1.15])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - 1 / 4 * np.array([-6.5, -6.5])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - 1 / 4 * np.array([6.5, -6.5])) < 1e-10)
