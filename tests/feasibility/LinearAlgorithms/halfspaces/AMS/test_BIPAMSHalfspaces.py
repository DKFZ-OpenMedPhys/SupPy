import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import BlockIterativeAMSHalfspace
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


@pytest.fixture
def get_BlockIterativeAMSHalfspace_input_full_sequential(get_full_variables):
    A, b = get_full_variables
    return BlockIterativeAMSHalfspace(A, b, weights=np.eye(8)), A, b


@pytest.fixture
def get_BlockIterativeAMSHalfspace_input_full_simultaneous(get_full_variables):
    A, b = get_full_variables
    return (
        BlockIterativeAMSHalfspace(
            A, b, weights=[[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]]
        ),
        A,
        b,
    )


@pytest.fixture
def get_BlockIterativeAMSHalfspace_input_sparse_sequential(get_sparse_variables):
    A, b = get_sparse_variables
    return BlockIterativeAMSHalfspace(A, b, weights=np.eye(8)), A, b


@pytest.fixture
def get_BlockIterativeAMSHalfspace_input_sparse_simultaneous(get_sparse_variables):
    A, b = get_sparse_variables
    return (
        BlockIterativeAMSHalfspace(
            A, b, weights=[[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]]
        ),
        A,
        b,
    )


def test_BlockIterativeAMSHalfspace_sequential_constructor_full(
    get_BlockIterativeAMSHalfspace_input_full_sequential,
):
    """
    Test the BlockIterativeAMSHalfspace constructor with sequential weights
    and a
    full matrix.
    """
    alg, A, b = get_BlockIterativeAMSHalfspace_input_full_sequential

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, [[1], [1], [1], [1], [1], [1], [1], [1]])
    assert np.array_equal(alg.idxs, np.eye(8))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeAMSHalfspace_simultaneous_constructor_full(
    get_BlockIterativeAMSHalfspace_input_full_simultaneous,
):
    """
    Test the BlockIterativeAMSHalfspace constructor with simultaneous
    weights and a full matrix.
    """
    alg, A, b = get_BlockIterativeAMSHalfspace_input_full_simultaneous

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, [[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]])
    assert np.array_equal(alg.idxs, [[True, True, True, True, True, True, True, True]])
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeAMSHalfspace_sequential_constructor_sparse(
    get_BlockIterativeAMSHalfspace_input_sparse_sequential,
):
    """
    Test the BlockIterativeAMSHalfspace constructor with sequential weights
    and a
    sparse matrix.
    """
    alg, A, b = get_BlockIterativeAMSHalfspace_input_sparse_sequential

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, [[1], [1], [1], [1], [1], [1], [1], [1]])
    assert np.array_equal(alg.idxs, np.eye(8))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeAMSHalfspace_simultaneous_constructor_sparse(
    get_BlockIterativeAMSHalfspace_input_sparse_simultaneous,
):
    """
    Test the BlockIterativeAMSHalfspace constructor with simultaneous
    weights and a
    sparse matrix.
    """
    alg, A, b = get_BlockIterativeAMSHalfspace_input_sparse_simultaneous

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, [[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]])
    assert np.array_equal(alg.idxs, [[True, True, True, True, True, True, True, True]])
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeAMSHalfspace_map_full(get_BlockIterativeAMSHalfspace_input_full_sequential):
    """
    Test the map function of the BlockIterativeAMSHalfspace class with
    sequential
    weights and full matrix.
    """
    alg, _, _ = get_BlockIterativeAMSHalfspace_input_full_sequential

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, 0, 1, 1, -2, 0, -1, -1]))


def test_BlockIterativeAMSHalfspace_map_sparse(
    get_BlockIterativeAMSHalfspace_input_sparse_sequential,
):
    """
    Test the map function of the BlockIterativeAMSHalfspace class with
    sequential
    weights and sparse matrix.
    """
    alg, _, _ = get_BlockIterativeAMSHalfspace_input_sparse_sequential

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, 0, 1, 1, -2, 0, -1, -1]))


def test_BlockIterativeAMSHalfspace_indexed_map_full(
    get_BlockIterativeAMSHalfspace_input_full_sequential,
):
    """
    Test the indexed_map function of the BlockIterativeAMSHalfspace class
    with
    sequential weights and full matrix.
    """
    alg, _, _ = get_BlockIterativeAMSHalfspace_input_full_sequential
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_BlockIterativeAMSHalfspace_indexed_map_sparse(
    get_BlockIterativeAMSHalfspace_input_sparse_sequential,
):
    """
    Test the indexed_map function of the BlockIterativeAMSHalfspace class
    with
    sequential weights and sparse matrix.
    """
    alg, _, _ = get_BlockIterativeAMSHalfspace_input_sparse_sequential
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_BlockIterativeAMSHalfspace_sequential_step_full(
    get_BlockIterativeAMSHalfspace_input_full_sequential,
):
    """
    Test the step function of the BlockIterativeAMSHalfspace class with
    sequential
    weights and full matrix.
    """
    alg, _, _ = get_BlockIterativeAMSHalfspace_input_full_sequential

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


def test_BlockIterativeAMSHalfspace_simultaneous_step_full(
    get_BlockIterativeAMSHalfspace_input_full_simultaneous,
):
    """
    Test the step function of the BlockIterativeAMSHalfspace class with
    simultaneous
    weights and full matrix.
    """
    alg, _, _ = get_BlockIterativeAMSHalfspace_input_full_simultaneous

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(np.abs(x_proj - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(np.abs(x_n - np.array([29 / 16, 29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(np.abs(x_n - np.array([-1.175, -1.175])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(np.abs(x_n + np.array([29 / 16, 29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(np.abs(x_n - np.array([29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_BlockIterativeAMSHalfspace_simultaneous_step_sparse(
    get_BlockIterativeAMSHalfspace_input_sparse_simultaneous,
):
    """
    Test the step function of the BlockIterativeAMSHalfspace class with
    simultaneous
    weights and sparse matrix.
    """
    alg, _, _ = get_BlockIterativeAMSHalfspace_input_sparse_simultaneous

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(np.abs(x_n - np.array([1.175, 1.175])) < 1e-10)

    x_n = alg.step(x_2)
    assert np.all(np.abs(x_n - np.array([29 / 16, 29 / 16])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(np.abs(x_n - np.array([-1.175, -1.175])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(np.abs(x_n + np.array([29 / 16, 29 / 16])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(np.abs(x_n - np.array([29 / 16, -29 / 16])) < 1e-10)
