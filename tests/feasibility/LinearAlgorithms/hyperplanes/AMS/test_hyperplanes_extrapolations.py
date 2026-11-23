import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import AdaptiveStepLandweberHyperplane, ExtrapolatedLandweberHyperplane


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
def get_ExtrapolatedLandweberHyperplane_input_full(get_full_variables):
    A, ub = get_full_variables
    return ExtrapolatedLandweberHyperplane(A, ub), A, ub


@pytest.fixture
def get_ExtrapolatedLandweberHyperplane_input_sparse(get_sparse_variables):
    A, ub = get_sparse_variables
    return ExtrapolatedLandweberHyperplane(A, ub), A, ub


@pytest.fixture
def get_AdaptiveStepLandweberHyperplane_input_full(get_full_variables):
    A, ub = get_full_variables
    return AdaptiveStepLandweberHyperplane(A, ub), A, ub


@pytest.fixture
def get_AdaptiveStepLandweberHyperplane_input_sparse(get_sparse_variables):
    A, ub = get_sparse_variables
    return AdaptiveStepLandweberHyperplane(A, ub), A, ub


def test_ExtrapolatedLandweberHyperplane_project(get_ExtrapolatedLandweberHyperplane_input_full):
    alg, A, ub = get_ExtrapolatedLandweberHyperplane_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.0, 1.0]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.0, 1.0]))


def test_AdaptiveLandweberHyperplane_project(get_AdaptiveStepLandweberHyperplane_input_full):
    alg, A, ub = get_AdaptiveStepLandweberHyperplane_input_full
    x = np.array([2.0, 2.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.0, 1.0]))

    x = np.array([0.0, 0.0])
    proj = alg.project(x)
    assert np.array_equal(proj, np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------
# ExtrapolatedLandweberHyperplane: options and branches
# ---------------------------------------------------------------------------


def test_ExtrapolatedLandweberHyperplane_constructor_state(get_full_variables):
    A, b = get_full_variables
    alg = ExtrapolatedLandweberHyperplane(A, b)
    # a_i are the squared row norms; weight_norm = weights / a_i
    assert np.allclose(alg.a_i, np.array([1.0, 0.5, 1.0]))
    assert np.allclose(alg.weight_norm, (1 / 3) / np.array([1.0, 0.5, 1.0]))
    assert alg.sigmas == []


def test_ExtrapolatedLandweberHyperplane_relaxation(get_full_variables):
    A, b = get_full_variables
    alg = ExtrapolatedLandweberHyperplane(A, b, relaxation=0.5)
    # the unrelaxed step goes exactly to the solution [1, 1]; the outer
    # relaxation halves it: [2, 2] + 0.5 * ([1, 1] - [2, 2]) = [1.5, 1.5]
    proj = alg.project(np.array([2.0, 2.0]))
    assert np.allclose(proj, np.array([1.5, 1.5]))


@pytest.mark.parametrize(
    "make_alg",
    [
        lambda A, b: ExtrapolatedLandweberHyperplane(A, b),
        lambda A, b: AdaptiveStepLandweberHyperplane(A, b),
    ],
    ids=["extrapolated", "adaptive_step"],
)
def test_Landweber_hyperplane_noop_on_solution(make_alg, get_full_variables):
    # on the exact solution all residuals vanish: the early-return branch
    # is taken and sigma = 0 is recorded
    A, b = get_full_variables
    alg = make_alg(A, b)
    x = np.array([1.0, 1.0])
    assert np.array_equal(alg.project(x.copy()), x)
    assert alg.sigmas == [0]


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_alg",
    [
        lambda A, b: ExtrapolatedLandweberHyperplane(A, b),
        lambda A, b: AdaptiveStepLandweberHyperplane(A, b),
    ],
    ids=["extrapolated", "adaptive_step"],
)
def test_Landweber_hyperplane_solve_converges(make_alg, get_full_variables):
    # the system is consistent with unique solution [1, 1]
    A, b = get_full_variables
    rng = np.random.default_rng(42)
    for _ in range(3):
        alg = make_alg(A, b)
        x = 4 * rng.standard_normal(2)
        x_n = alg.solve(x, max_iter=500)
        # prox_tol=1e-6 with weights 1/3 only guarantees residuals ~1.7e-3
        assert np.allclose(x_n, np.array([1.0, 1.0]), atol=1e-2)
        assert alg.proximities[-1][0] < 1e-6


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_Landweber_hyperplane_solve_int_input(get_full_variables):
    A, b = get_full_variables
    alg = ExtrapolatedLandweberHyperplane(A, b)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_Landweber_hyperplane_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    for make_alg in (ExtrapolatedLandweberHyperplane, AdaptiveStepLandweberHyperplane):
        x_np = make_alg(A, b).project(np.array([2.0, 2.0]))
        alg_cp = make_alg(cp.asarray(A), cp.asarray(b))
        x_cp = alg_cp.project(cp.asarray([2.0, 2.0]))
        assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
        assert np.allclose(cp.asnumpy(x_cp), x_np)
