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
    assert np.array_equal(alg.weights, np.array([[1], [1], [1], [1], [1], [1], [1], [1]]))
    assert np.array_equal(alg.block_idxs, np.array([[0], [1], [2], [3], [4], [5], [6], [7]]))
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
    assert np.array_equal(
        alg.weights, np.array([[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]])
    )
    assert np.array_equal(alg.block_idxs, [[0, 1, 2, 3, 4, 5, 6, 7]])
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
    assert np.array_equal(alg.weights, np.array([[1], [1], [1], [1], [1], [1], [1], [1]]))
    assert np.array_equal(alg.block_idxs, np.array([[0], [1], [2], [3], [4], [5], [6], [7]]))
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
    assert np.array_equal(
        alg.weights, np.array([[1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8, 1 / 8]])
    )
    assert np.array_equal(alg.block_idxs, [[0, 1, 2, 3, 4, 5, 6, 7]])
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
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([0, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([0, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0])) < 1e-10)
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
    assert np.all(abs(x_n + np.array([29 / 16, 29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([29 / 16, -29 / 16])) < 1e-10)
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
    assert np.all(abs(x_n - np.array([1.175, 1.175])) < 1e-10)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([29 / 16, 29 / 16])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.175, -1.175])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n + np.array([29 / 16, 29 / 16])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([29 / 16, -29 / 16])) < 1e-10)


def test_BlockIterativeAMSHalfspace_proximity_sequential(
    get_BlockIterativeAMSHalfspace_input_full_sequential,
):
    alg, _, _ = get_BlockIterativeAMSHalfspace_input_full_sequential
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


def test_BlockIterativeAMSHalfspace_proximity_simultaneous(
    get_BlockIterativeAMSHalfspace_input_full_simultaneous,
):

    alg, _, _ = get_BlockIterativeAMSHalfspace_input_full_simultaneous
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


# ---------------------------------------------------------------------------
# BlockIterativeAMSHalfspace: options and branches
# ---------------------------------------------------------------------------


def test_BlockIterativeAMSHalfspace_weights_row_sum_error(get_full_variables):
    A, b = get_full_variables
    # every block's weights must sum to 1
    with pytest.raises(ValueError):
        BlockIterativeAMSHalfspace(
            A,
            b,
            weights=[
                np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]),
                np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.35]),
            ],
        )


def test_BlockIterativeAMSHalfspace_two_block_state(get_full_variables):
    A, b = get_full_variables
    weights = [
        np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]),
    ]
    alg = BlockIterativeAMSHalfspace(A, b, weights=weights)
    # nonzero entries define the blocks; the stored weights are filtered
    assert [list(blk) for blk in alg.block_idxs] == [[0, 1, 2, 3], [4, 5, 6, 7]]
    assert all(np.array_equal(w, np.array([0.25] * 4)) for w in alg.weights)
    # total_weights averages the blocks: ([.25]*4+[0]*4 + [0]*4+[.25]*4) / 2
    assert np.allclose(alg.total_weights, np.full(8, 0.125))


def test_BlockIterativeAMSHalfspace_proximity_flag_false(get_full_variables):
    A, b = get_full_variables
    alg = BlockIterativeAMSHalfspace(A, b, weights=np.eye(8), proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_BlockIterativeAMSHalfspace_eye_matches_sequential(get_full_variables):
    # one block per constraint (identity weights) is exactly the sequential
    # sweep
    from suppy.feasibility import SequentialAMSHalfspace

    A, b = get_full_variables
    alg_b = BlockIterativeAMSHalfspace(A, b, weights=np.eye(8))
    alg_s = SequentialAMSHalfspace(A, b)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_b.project(x.copy()), alg_s.project(x.copy()))


def test_BlockIterativeAMSHalfspace_single_block_matches_simultaneous(get_full_variables):
    # a single uniform block is exactly the simultaneous step
    from suppy.feasibility import SimultaneousAMSHalfspace

    A, b = get_full_variables
    alg_b = BlockIterativeAMSHalfspace(A, b, weights=np.array([[1 / 8] * 8]))
    alg_s = SimultaneousAMSHalfspace(A, b)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_b.project(x.copy()), alg_s.project(x.copy()))


def test_BlockIterativeAMSHalfspace_solve_reduces_proximity(get_full_variables):
    A, b = get_full_variables
    weights = [
        np.array([0.25, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.25, 0.25]),
    ]
    alg = BlockIterativeAMSHalfspace(A, b, weights=weights)
    alg.solve(np.array([2.0, 2.0]), max_iter=200)
    assert alg.proximities[-1][0] < alg.proximities[0][0]
    assert alg.proximities[-1][0] < 1e-6


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_BlockIterativeAMSHalfspace_solve_int_input(get_full_variables):
    A, b = get_full_variables
    alg = BlockIterativeAMSHalfspace(A, b, weights=np.eye(8))
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_BlockIterativeAMSHalfspace_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    w_np = np.array([[1 / 8] * 8])
    x_np = BlockIterativeAMSHalfspace(A, b, weights=w_np).step(np.array([2.0, 2.0]))

    alg_cp = BlockIterativeAMSHalfspace(cp.asarray(A), cp.asarray(b), weights=cp.asarray(w_np))
    x_cp = alg_cp.step(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
