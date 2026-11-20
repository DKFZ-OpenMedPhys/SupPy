import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import ExtrapolatedLandweberHalfspace, AdaptiveStepLandweberHalfspace


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
def get_ExtrapolatedLandweberHalfspace_input_full(get_full_variables):
    A, ub = get_full_variables
    return ExtrapolatedLandweberHalfspace(A, ub), A, ub


@pytest.fixture
def get_ExtrapolatedLandweberHalfspace_input_sparse(get_sparse_variables):
    A, ub = get_sparse_variables
    return ExtrapolatedLandweberHalfspace(A, ub), A, ub


@pytest.fixture
def get_AdaptiveStepLandweberHalfspace_input_full(get_full_variables):
    A, ub = get_full_variables
    return AdaptiveStepLandweberHalfspace(A, ub), A, ub


@pytest.fixture
def get_AdaptiveStepLandweberHalfspace_input_sparse(get_sparse_variables):
    A, ub = get_sparse_variables
    return AdaptiveStepLandweberHalfspace(A, ub), A, ub


def test_ExtrapolatedLandweberHalfspace_project(get_ExtrapolatedLandweberHalfspace_input_full):

    alg, A, ub = get_ExtrapolatedLandweberHalfspace_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_ExtrapolatedLandweberHalfspace_project_sparse(
    get_ExtrapolatedLandweberHalfspace_input_sparse,
):

    alg, A, ub = get_ExtrapolatedLandweberHalfspace_input_sparse
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([7 / 6, 7 / 6]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_AdaptiveStepLandweberHalfspace_project(get_AdaptiveStepLandweberHalfspace_input_full):

    alg, A, ub = get_AdaptiveStepLandweberHalfspace_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    np.testing.assert_allclose(proj, np.array([19 / 12, 19 / 12]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([0.0, 0.0]))


# ---------------------------------------------------------------------------
# ExtrapolatedLandweberHalfspace: options and branches
# ---------------------------------------------------------------------------


def test_ExtrapolatedLandweberHalfspace_constructor_state(get_full_variables):
    A, b = get_full_variables
    alg = ExtrapolatedLandweberHalfspace(A, b)
    # a_i are the squared row norms; weight_norm = weights / a_i
    assert np.allclose(alg.a_i, np.array([2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]))
    assert np.allclose(alg.weight_norm, 0.125 / np.array([2.0, 2.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0]))
    assert alg.sigmas == []


def test_ExtrapolatedLandweberHalfspace_zero_row():
    # an all-zero row gets weight_norm 0 in the constructor; if that row's
    # halfspace is the only violated one the step degenerates to 0/0, which the
    # step size guard turns into no step (as for the hyperslab variant)
    A = np.array([[1.0, 1.0], [0.0, 0.0]])
    b = np.array([2.0, -1.0])  # row 1: 0 <= -1 is never satisfiable
    alg = ExtrapolatedLandweberHalfspace(A, b)
    assert alg.weight_norm[1] == 0

    proj = alg.project(np.array([0.0, 0.0]))
    assert np.array_equal(proj, np.array([0.0, 0.0]))
    assert alg.sigmas[-1] == 0


def test_ExtrapolatedLandweberHalfspace_relaxation(get_full_variables):
    A, b = get_full_variables
    alg = ExtrapolatedLandweberHalfspace(A, b, relaxation=0.5)
    # the outer relaxation halves the step: x + 0.5 * ([7/6, 7/6] - [2, 2])
    proj = alg.project(np.array([2.0, 2.0]))
    assert np.allclose(proj, np.array([19 / 12, 19 / 12]))


def test_ExtrapolatedLandweberHalfspace_sigma_zero_on_feasible(
    get_ExtrapolatedLandweberHalfspace_input_full,
):
    # feasible points take the early-return branch and record sigma = 0
    alg, _, _ = get_ExtrapolatedLandweberHalfspace_input_full
    alg.project(np.array([0.5, 0.5]))
    assert alg.sigmas == [0]


def test_ExtrapolatedLandweberHalfspace_proximity(get_ExtrapolatedLandweberHalfspace_input_full):
    # the proximity is inherited from SimultaneousAMSHalfspace (weights 1/8):
    # x = [2, 2] -> violations [2, 0.5, 0.5] -> 1/8 * 4.5 = 9/16
    alg, _, _ = get_ExtrapolatedLandweberHalfspace_input_full
    prox = alg.proximity(np.array([2.0, 2.0]), [("p_norm", 2), "max_norm"])
    assert np.abs(prox[0] - 9 / 16) < 1e-10
    assert np.abs(prox[1] - 2.0) < 1e-10


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_alg",
    [
        lambda A, b: ExtrapolatedLandweberHalfspace(A, b),
        lambda A, b: AdaptiveStepLandweberHalfspace(A, b),
    ],
    ids=["extrapolated", "adaptive_step"],
)
def test_Landweber_halfspace_solve_converges(make_alg, get_full_variables):
    A, b = get_full_variables
    alg = make_alg(A, b)
    rng = np.random.default_rng(42)
    for _ in range(3):
        x = 4 * rng.standard_normal(2)
        alg.solve(x, max_iter=200)
        assert alg.proximities[-1][0] < alg.proximities[0][0]
        assert alg.proximities[-1][0] < 1e-6


@pytest.mark.parametrize(
    "make_alg",
    [
        lambda A, b: ExtrapolatedLandweberHalfspace(A, b),
        lambda A, b: AdaptiveStepLandweberHalfspace(A, b),
    ],
    ids=["extrapolated", "adaptive_step"],
)
def test_Landweber_halfspace_project_idempotent_on_feasible(make_alg, get_full_variables):
    # feasible points take the early-return branch and stay untouched
    A, b = get_full_variables
    alg = make_alg(A, b)
    x = np.array([0.5, 0.5])
    assert np.array_equal(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_Landweber_halfspace_solve_int_input(get_full_variables):
    A, b = get_full_variables
    alg = ExtrapolatedLandweberHalfspace(A, b)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_Landweber_halfspace_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    for make_alg in (ExtrapolatedLandweberHalfspace, AdaptiveStepLandweberHalfspace):
        x_np = make_alg(A, b).project(np.array([2.0, 2.0]))
        alg_cp = make_alg(cp.asarray(A), cp.asarray(b))
        x_cp = alg_cp.project(cp.asarray([2.0, 2.0]))
        assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
        assert np.allclose(cp.asnumpy(x_cp), x_np)
