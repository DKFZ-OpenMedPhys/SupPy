import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import (
    ExtrapolatedLandweberHyperslab,
    AdaptiveStepLandweberHyperslab,
    BlockIterativeAdaptiveStepLandweberHyperslab,
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


@pytest.fixture
def get_ExtrapolatedLandweberHyperslab_input_full(get_full_variables):
    A, lb, ub = get_full_variables
    return ExtrapolatedLandweberHyperslab(A, lb, ub), A, lb, ub


@pytest.fixture
def get_ExtrapolatedLandweberHyperslab_input_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return ExtrapolatedLandweberHyperslab(A, lb, ub), A, lb, ub


@pytest.fixture
def get_AdaptiveStepLandweberHyperslab_input_full(get_full_variables):
    A, lb, ub = get_full_variables
    return AdaptiveStepLandweberHyperslab(A, lb, ub), A, lb, ub


@pytest.fixture
def get_AdaptiveStepLandweberHyperslab_input_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return AdaptiveStepLandweberHyperslab(A, lb, ub), A, lb, ub


def test_ExtrapolatedLandweberHyperslab_project(get_ExtrapolatedLandweberHyperslab_input_full):
    alg, A, lb, ub = get_ExtrapolatedLandweberHyperslab_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_ExtrapolatedLandweberHyperslab_project_sparse(
    get_ExtrapolatedLandweberHyperslab_input_sparse,
):
    alg, A, lb, ub = get_ExtrapolatedLandweberHyperslab_input_sparse
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_AdaptiveStepLandweberHyperslab_project(get_AdaptiveStepLandweberHyperslab_input_full):
    alg, A, lb, ub = get_AdaptiveStepLandweberHyperslab_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_AdaptiveStepLandweberHyperslab_project_sparse(
    get_AdaptiveStepLandweberHyperslab_input_sparse,
):
    alg, A, lb, ub = get_AdaptiveStepLandweberHyperslab_input_sparse
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_BlockIterativeAdaptiveStepLandweberHyperslab_single_block(get_full_variables):
    # with a single block the algorithm is identical to the simultaneous version
    A, lb, ub = get_full_variables
    alg = BlockIterativeAdaptiveStepLandweberHyperslab(A, lb, ub, length_blocks=[4])
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.allclose(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_BlockIterativeAdaptiveStepLandweberHyperslab_project(get_full_variables):
    A, lb, ub = get_full_variables
    alg = BlockIterativeAdaptiveStepLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])
    # block 1 (rows 0, 1): only row 0 is violated, u = (-0.5, -0.5), sig = 2 -> x = (1, 1)
    # block 2 (rows 2, 3): feasible, no step
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.allclose(proj, np.array([1.0, 1.0]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_BlockIterativeAdaptiveStepLandweberHyperslab_project_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    alg = BlockIterativeAdaptiveStepLandweberHyperslab(A, lb, ub, length_blocks=[2, 2])
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.allclose(proj, np.array([1.0, 1.0]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# ExtrapolatedLandweberHyperslab: options and branches
# ---------------------------------------------------------------------------


def test_ExtrapolatedLandweberHyperslab_constructor_state(get_full_variables):
    A, lb, ub = get_full_variables
    alg = ExtrapolatedLandweberHyperslab(A, lb, ub)
    # a_i are the squared row norms; weight_norm = weights / a_i
    assert np.allclose(alg.a_i, np.array([2.0, 2.0, 1.0, 1.0]))
    assert np.allclose(alg.weight_norm, 0.25 / np.array([2.0, 2.0, 1.0, 1.0]))


def test_ExtrapolatedLandweberHyperslab_zero_row():
    # an all-zero row gets weight_norm 0 in the constructor; if that row's
    # band is the only violated one the step degenerates to 0/0 and the NaN
    # guard raises (current contract, cf. the TODO in _ams_extrapolations.py)
    A = np.array([[1.0, 1.0], [0.0, 0.0]])
    lb = np.array([-2.0, 1.0])
    ub = np.array([2.0, 2.0])
    alg = ExtrapolatedLandweberHyperslab(A, lb, ub)
    assert alg.weight_norm[1] == 0

    with np.errstate(invalid="ignore"):
        with pytest.raises(ValueError, match="NaN"):
            alg.project(np.array([0.0, 0.0]))


def test_ExtrapolatedLandweberHyperslab_relaxation(get_full_variables):
    A, lb, ub = get_full_variables
    alg = ExtrapolatedLandweberHyperslab(A, lb, ub, relaxation=0.5)
    # the outer relaxation halves the step: x + 0.5 * ([7/6, 7/6] - [2, 2])
    proj = alg.project(np.array([2.0, 2.0]))
    assert np.allclose(proj, np.array([19 / 12, 19 / 12]))


def test_ExtrapolatedLandweberHyperslab_proximity(get_ExtrapolatedLandweberHyperslab_input_full):
    # the proximity is inherited from SimultaneousAMSHyperslab (weights 1/4):
    # x = [2, 2] -> residuals [2, 0, 0.5, 0.5] -> 1/4 * 4.5 = 9/8
    alg, _, _, _ = get_ExtrapolatedLandweberHyperslab_input_full
    prox = alg.proximity(np.array([2.0, 2.0]), [("p_norm", 2), "max_norm"])
    assert np.abs(prox[0] - 9 / 8) < 1e-10
    assert np.abs(prox[1] - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_alg",
    [
        lambda A, lb, ub: ExtrapolatedLandweberHyperslab(A, lb, ub),
        lambda A, lb, ub: AdaptiveStepLandweberHyperslab(A, lb, ub),
        lambda A, lb, ub: BlockIterativeAdaptiveStepLandweberHyperslab(
            A, lb, ub, length_blocks=[2, 2]
        ),
    ],
    ids=["extrapolated", "adaptive_step", "block_iterative_adaptive"],
)
def test_Landweber_solve_converges(make_alg, get_full_variables):
    A, lb, ub = get_full_variables
    alg = make_alg(A, lb, ub)
    rng = np.random.default_rng(42)
    for _ in range(3):
        x = 4 * rng.standard_normal(2)
        alg.solve(x, max_iter=200)
        assert alg.proximities[-1][0] < alg.proximities[0][0]
        assert alg.proximities[-1][0] < 1e-6


@pytest.mark.parametrize(
    "make_alg",
    [
        lambda A, lb, ub: ExtrapolatedLandweberHyperslab(A, lb, ub),
        lambda A, lb, ub: AdaptiveStepLandweberHyperslab(A, lb, ub),
    ],
    ids=["extrapolated", "adaptive_step"],
)
def test_Landweber_project_idempotent_on_feasible(make_alg, get_full_variables):
    # feasible points take the early-return branch and stay untouched
    A, lb, ub = get_full_variables
    alg = make_alg(A, lb, ub)
    x = np.array([0.5, 0.5])
    assert np.array_equal(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_Landweber_solve_int_input(get_full_variables):
    A, lb, ub = get_full_variables
    alg = ExtrapolatedLandweberHyperslab(A, lb, ub)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_Landweber_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_full_variables
    for make_alg in (ExtrapolatedLandweberHyperslab, AdaptiveStepLandweberHyperslab):
        x_np = make_alg(A, lb, ub).project(np.array([2.0, 2.0]))
        alg_cp = make_alg(cp.asarray(A), cp.asarray(lb), cp.asarray(ub))
        x_cp = alg_cp.project(cp.asarray([2.0, 2.0]))
        assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
        assert np.allclose(cp.asnumpy(x_cp), x_np)
