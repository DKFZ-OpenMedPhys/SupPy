import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import BlockIterativeKaczmarz
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


@pytest.fixture
def get_BlockIterativeKaczmarz_input_full_sequential(get_full_variables):
    A, b = get_full_variables
    return BlockIterativeKaczmarz(A, b, weights=np.eye(3)), A, b


@pytest.fixture
def get_BlockIterativeKaczmarz_input_full_simultaneous(get_full_variables):
    A, b = get_full_variables
    return (
        BlockIterativeKaczmarz(A, b, weights=[[1 / 3, 1 / 3, 1 / 3]]),
        A,
        b,
    )


@pytest.fixture
def get_BlockIterativeKaczmarz_input_sparse_sequential(get_sparse_variables):
    A, b = get_sparse_variables
    return BlockIterativeKaczmarz(A, b, weights=np.eye(3)), A, b


@pytest.fixture
def get_BlockIterativeKaczmarz_input_sparse_simultaneous(get_sparse_variables):
    A, b = get_sparse_variables
    return (
        BlockIterativeKaczmarz(A, b, weights=[[1 / 3, 1 / 3, 1 / 3]]),
        A,
        b,
    )


def test_BlockIterativeKaczmarz_sequential_constructor_full(
    get_BlockIterativeKaczmarz_input_full_sequential,
):
    """
    Test the BlockIterativeKaczmarz constructor with sequential weights
    and a
    full matrix.
    """
    alg, A, b = get_BlockIterativeKaczmarz_input_full_sequential

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, [[1], [1], [1]])
    assert np.array_equal(alg.block_idxs, np.array([[0], [1], [2]]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeKaczmarz_simultaneous_constructor_full(
    get_BlockIterativeKaczmarz_input_full_simultaneous,
):
    """
    Test the BlockIterativeKaczmarz constructor with simultaneous
    weights and a full matrix.
    """
    alg, A, b = get_BlockIterativeKaczmarz_input_full_simultaneous

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, [[1 / 3, 1 / 3, 1 / 3]])
    assert np.array_equal(alg.block_idxs, np.array([[0, 1, 2]]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeKaczmarz_sequential_constructor_sparse(
    get_BlockIterativeKaczmarz_input_sparse_sequential,
):
    """
    Test the BlockIterativeKaczmarz constructor with sequential weights
    and a
    sparse matrix.
    """
    alg, A, b = get_BlockIterativeKaczmarz_input_sparse_sequential

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, [[1], [1], [1]])
    assert np.array_equal(alg.block_idxs, np.array([[0], [1], [2]]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeKaczmarz_simultaneous_constructor_sparse(
    get_BlockIterativeKaczmarz_input_sparse_simultaneous,
):
    """
    Test the BlockIterativeKaczmarz constructor with simultaneous
    weights and a
    sparse matrix.
    """
    alg, A, b = get_BlockIterativeKaczmarz_input_sparse_simultaneous

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, [[1 / 3, 1 / 3, 1 / 3]])
    assert np.array_equal(alg.block_idxs, np.array([[0, 1, 2]]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_BlockIterativeKaczmarz_map_full(
    get_BlockIterativeKaczmarz_input_full_sequential,
):
    """
    Test the map function of the BlockIterativeKaczmarz class with
    sequential
    weights and full matrix.
    """
    alg, _, _ = get_BlockIterativeKaczmarz_input_full_sequential

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([1, 0, 1]))


def test_BlockIterativeKaczmarz_map_sparse(
    get_BlockIterativeKaczmarz_input_sparse_sequential,
):
    """
    Test the map function of the BlockIterativeKaczmarz class with
    sequential
    weights and sparse matrix.
    """
    alg, _, _ = get_BlockIterativeKaczmarz_input_sparse_sequential

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([1, 0, 1]))


def test_BlockIterativeKaczmarz_indexed_map_full(
    get_BlockIterativeKaczmarz_input_full_sequential,
):
    """
    Test the indexed_map function of the BlockIterativeKaczmarz class
    with
    sequential weights and full matrix.
    """
    alg, _, _ = get_BlockIterativeKaczmarz_input_full_sequential
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([1, 0]))


def test_BlockIterativeKaczmarz_indexed_map_sparse(
    get_BlockIterativeKaczmarz_input_sparse_sequential,
):
    """
    Test the indexed_map function of the BlockIterativeKaczmarz class
    with sequential weights and sparse matrix.
    """
    alg, _, _ = get_BlockIterativeKaczmarz_input_sparse_sequential
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([1, 0]))


def test_BlockIterativeKaczmarz_sequential_step_full(
    get_BlockIterativeKaczmarz_input_full_sequential,
):
    """
    Test the step function of the BlockIterativeKaczmarz class with
    sequential weights and full matrix.
    """
    alg, _, _ = get_BlockIterativeKaczmarz_input_full_sequential

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([2, 1]) < 1e-10))
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([2, 1]) < 1e-10))
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([1.5, 1]) < 1e-10))
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0.5, 1]) < 1e-10))
    assert np.array_equal(x_n, x_5)


def test_BlockIterativeKaczmarz_simultaneous_step_full(
    get_BlockIterativeKaczmarz_input_full_simultaneous,
):
    """
    Test the step function of the BlockIterativeKaczmarz class with
    simultaneous
    weights and full matrix.
    """
    alg, _, _ = get_BlockIterativeKaczmarz_input_full_simultaneous

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


def test_BlockIterativeKaczmarz_simultaneous_step_sparse(
    get_BlockIterativeKaczmarz_input_sparse_simultaneous,
):
    """
    Test the step function of the BlockIterativeKaczmarz class with
    simultaneous
    weights and sparse matrix.
    """
    alg, _, _ = get_BlockIterativeKaczmarz_input_sparse_simultaneous

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


def test_BlockIterativeKaczmarz_proximity_sequential(
    get_BlockIterativeKaczmarz_input_full_sequential,
):

    alg, _, _ = get_BlockIterativeKaczmarz_input_full_sequential

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    prox_measures = []
    no_prox = alg.proximity(x_1, prox_measures)
    assert no_prox.size == 0

    prox_measures = [("p_norm", 2), "max_norm"]
    prox_1 = alg.proximity(x_1, prox_measures)
    prox_2 = alg.proximity(x_2, prox_measures)
    prox_3 = alg.proximity(x_3, prox_measures)
    prox_4 = alg.proximity(x_4, prox_measures)
    prox_5 = alg.proximity(x_5, prox_measures)

    assert np.abs(prox_1[0] - 2 / 3) < 1e-10
    assert np.abs(prox_1[1] - 1) < 1e-10

    assert np.abs(prox_2[0] - 5 / 3) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 29 / 12) < 1e-10
    assert np.abs(prox_3[1] - 2) < 1e-10

    assert np.abs(prox_4[0] - 14 / 3) < 1e-10
    assert np.abs(prox_4[1] - 3) < 1e-10

    assert np.abs(prox_5[0] - 77 / 12) < 1e-10
    assert np.abs(prox_5[1] - 4) < 1e-10


def test_BlockIterativeKaczmarz_proximity_simultaneous(
    get_BlockIterativeKaczmarz_input_full_simultaneous,
):

    alg, _, _ = get_BlockIterativeKaczmarz_input_full_simultaneous

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    prox_measures = []
    no_prox = alg.proximity(x_1, prox_measures)
    assert no_prox.size == 0

    prox_measures = [("p_norm", 2), "max_norm"]
    prox_1 = alg.proximity(x_1, prox_measures)
    prox_2 = alg.proximity(x_2, prox_measures)
    prox_3 = alg.proximity(x_3, prox_measures)
    prox_4 = alg.proximity(x_4, prox_measures)
    prox_5 = alg.proximity(x_5, prox_measures)

    assert np.abs(prox_1[0] - 2 / 3) < 1e-10
    assert np.abs(prox_1[1] - 1) < 1e-10

    assert np.abs(prox_2[0] - 5 / 3) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 29 / 12) < 1e-10
    assert np.abs(prox_3[1] - 2) < 1e-10

    assert np.abs(prox_4[0] - 14 / 3) < 1e-10
    assert np.abs(prox_4[1] - 3) < 1e-10

    assert np.abs(prox_5[0] - 77 / 12) < 1e-10
    assert np.abs(prox_5[1] - 4) < 1e-10
