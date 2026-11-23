import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SARTHyperslab, SimultaneousAMSHyperslab
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


# ---------------------------------------------------------------------------
# SARTHyperslab: options and branches
# ---------------------------------------------------------------------------


def test_SARTHyperslab_constructor_full(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0
    # inherited simultaneous weights default to 1/m
    assert np.allclose(alg.weights, np.ones(4) / 4)
    # SART normalizations: 1-norms of the rows [2, 2, 1, 1] and columns [3, 3]
    assert np.allclose(alg.inverse_linear_row_norm, 1 / np.array([2.0, 2.0, 1.0, 1.0]))
    assert np.allclose(alg.inverse_linear_column_norm, 1 / np.array([3.0, 3.0]))


def test_SARTHyperslab_constructor_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    alg = SARTHyperslab(A, lb, ub)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.allclose(alg.inverse_linear_row_norm, 1 / np.array([2.0, 2.0, 1.0, 1.0]))
    assert np.allclose(alg.inverse_linear_column_norm, 1 / np.array([3.0, 3.0]))


def test_SARTHyperslab_project_full(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub)
    # x = [2, 2]: violated rows 0, 2, 3 with upper residuals [-2, -0.5, -0.5];
    # weighted update (weights 1/4, inverse linear row norms [1/2, 1/2, 1, 1]):
    #   0.25*0.5*(-2)*[1,1] + 0.25*1*(-0.5)*[1,0] + 0.25*1*(-0.5)*[0,1]
    #   = [-0.375, -0.375]
    # scaled by the inverse column norms 1/3 -> x - [0.125, 0.125] = [15/8, 15/8]
    x_n = alg.project(np.array([2.0, 2.0]))
    assert np.all(abs(x_n - np.array([15 / 8, 15 / 8])) < 1e-10)


def test_SARTHyperslab_project_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    alg = SARTHyperslab(A, lb, ub)
    x_n = alg.project(np.array([2.0, 2.0]))
    assert np.all(abs(x_n - np.array([15 / 8, 15 / 8])) < 1e-10)


def test_SARTHyperslab_algorithmic_relaxation(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub, algorithmic_relaxation=2.0)
    # doubles the update of the unrelaxed test: [2, 2] - [0.25, 0.25]
    x_n = alg.project(np.array([2.0, 2.0]))
    assert np.all(abs(x_n - np.array([1.75, 1.75])) < 1e-10)


def test_SARTHyperslab_custom_weights(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub, weights=np.array([0.4, 0.2, 0.2, 0.2]))
    # update = 1/3 * (0.4*0.5*(-2)*[1,1] + 0.2*1*(-0.5)*[1,0] + 0.2*1*(-0.5)*[0,1])
    #        = 1/3 * [-0.5, -0.5]
    x_n = alg.project(np.array([2.0, 2.0]))
    assert np.all(abs(x_n - (np.array([2.0, 2.0]) - 0.5 / 3)) < 1e-10)


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_SARTHyperslab_project_idempotent_on_feasible(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub)
    x = np.array([0.5, 0.5])
    assert np.array_equal(alg.project(x.copy()), x)


def test_SARTHyperslab_proximity_matches_simultaneous(get_full_variables):
    # the proximity is inherited unchanged from SimultaneousAMSHyperslab
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub)
    ref = SimultaneousAMSHyperslab(A, lb, ub)
    measures = [("p_norm", 2), "max_norm"]
    for x in (np.array([2.0, 2.0]), np.array([1.2, 1.2]), np.array([-3.0, 0.0])):
        assert np.allclose(alg.proximity(x, measures), ref.proximity(x, measures))


def test_SARTHyperslab_solve_reduces_proximity(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub)
    rng = np.random.default_rng(42)
    for _ in range(3):
        x = 4 * rng.standard_normal(2)
        alg.solve(x, max_iter=2000)
        assert alg.proximities[-1][0] < alg.proximities[0][0]
        assert alg.proximities[-1][0] < 1e-6


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SARTHyperslab_solve_int_input(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=2000)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_SARTHyperslab_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_full_variables
    x_np = SARTHyperslab(A, lb, ub).project(np.array([2.0, 2.0]))

    alg_cp = SARTHyperslab(cp.asarray(A), cp.asarray(lb), cp.asarray(ub))
    x_cp = alg_cp.project(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------


def test_SARTHyperslab_solve_to_feasibility(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SARTHyperslab(A, lb, ub)
    # prox_tol bounds the *mean squared* violation, so a per-row tolerance t
    # requires prox_tol ~ t^2 / m; 1e-12 allows row violations of ~2e-6
    x_n = alg.solve(np.array([2.0, 2.0]), max_iter=5000, prox_tol=1e-12, del_prox_tol=1e-14)
    p = A @ x_n
    assert np.all(p >= lb - 1e-4)
    assert np.all(p <= ub + 1e-4)
