import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import (
    BlockIterativeExtrapolatedLandweberHyperslab,
    ExtrapolatedLandweberHyperslab,
)


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


# ---------------------------------------------------------------------------
# BlockIterativeExtrapolatedLandweberHyperslab: options and branches
# ---------------------------------------------------------------------------


def test_BlockIterativeExtrapolatedLandweberHyperslab_constructor_state(get_full_variables):
    A, lb, ub = get_full_variables
    alg = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])

    assert alg.length_blocks == [2, 2]
    # the blocks slice the rows of A, the bounds and the weight norms
    assert np.array_equal(alg.As[0], A[:2])
    assert np.array_equal(alg.As[1], A[2:])
    assert np.array_equal(alg.ls[0], lb[:2])
    assert np.array_equal(alg.us[1], ub[2:])
    # weight_norm = weights / squared row norms = (1/4) / [2, 2, 1, 1]
    assert np.allclose(alg.weight_norms[0], np.array([1 / 8, 1 / 8]))
    assert np.allclose(alg.weight_norms[1], np.array([1 / 4, 1 / 4]))


def test_BlockIterativeExtrapolatedLandweberHyperslab_project(get_full_variables):
    A, lb, ub = get_full_variables
    alg = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])
    # block 1 (rows 0, 1): only row 0 violated with residual -2;
    # t_u = 1/8*(-2) = -0.25, t_u_2 = [-0.25, -0.25],
    # sig = ((-2)*(-0.25)) / 0.125 = 4 -> x += 4*[-0.25, -0.25] = [1, 1]
    # block 2 (rows 2, 3): feasible, continue
    x_n = alg.project(np.array([2.0, 2.0]))
    assert np.allclose(x_n, np.array([1.0, 1.0]))

    # feasible x takes the continue branch in every block
    x = np.array([0.0, 0.0])
    assert np.array_equal(alg.project(x), np.array([0.0, 0.0]))


def test_BlockIterativeExtrapolatedLandweberHyperslab_project_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    alg = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])
    x_n = alg.project(np.array([2.0, 2.0]))
    assert np.allclose(x_n, np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_BlockIterativeExtrapolatedLandweberHyperslab_single_block_matches_extrapolated(
    get_full_variables,
):
    # with a single block the algorithm is identical to the unblocked variant
    A, lb, ub = get_full_variables
    alg_b = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[4])
    alg_e = ExtrapolatedLandweberHyperslab(A, lb, ub)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_b.project(x.copy()), alg_e.project(x.copy()))


def test_BlockIterativeExtrapolatedLandweberHyperslab_project_idempotent_on_feasible(
    get_full_variables,
):
    A, lb, ub = get_full_variables
    alg = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])
    x = np.array([0.5, 0.5])
    assert np.array_equal(alg.project(x.copy()), x)


def test_BlockIterativeExtrapolatedLandweberHyperslab_solve_reduces_proximity(
    get_full_variables,
):
    A, lb, ub = get_full_variables
    alg = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])
    rng = np.random.default_rng(42)
    for _ in range(3):
        x = 4 * rng.standard_normal(2)
        alg.solve(x, max_iter=200)
        assert alg.proximities[-1][0] < alg.proximities[0][0]
        assert alg.proximities[-1][0] < 1e-6


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_BlockIterativeExtrapolatedLandweberHyperslab_solve_int_input(get_full_variables):
    A, lb, ub = get_full_variables
    alg = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_BlockIterativeExtrapolatedLandweberHyperslab_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_full_variables
    x_np = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[2, 2]).project(
        np.array([2.0, 2.0])
    )

    alg_cp = BlockIterativeExtrapolatedLandweberHyperslab(
        cp.asarray(A), cp.asarray(lb), cp.asarray(ub), length_blocks=[2, 2]
    )
    x_cp = alg_cp.project(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------


def test_BlockIterativeExtrapolatedLandweberHyperslab_solve_to_feasibility(get_full_variables):
    A, lb, ub = get_full_variables
    alg = BlockIterativeExtrapolatedLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])
    x_n = alg.solve(np.array([2.0, 2.0]), max_iter=200)
    p = A @ x_n
    assert np.all(p >= lb - 1e-3)
    assert np.all(p <= ub + 1e-3)
