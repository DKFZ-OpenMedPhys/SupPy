import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import (
    LandweberHyperplane,
    LandweberHalfspace,
    LandweberHyperslab,
    SimultaneousKaczmarzMethod,
)


@pytest.fixture
def A():
    return np.array([[1.0, 0.0], [1 / 2, -1 / 2], [0.0, 1.0], [2.0, 1.0]])


@pytest.fixture
def x0():
    return np.array([3.0, -2.0])


# ---------------------------------------------------------------------------
# step size
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sparse_A", [False, True])
def test_default_step_size(A, sparse_A):
    L = np.linalg.norm(A, 2) ** 2
    A_in = sparse.csr_array(A) if sparse_A else A
    b = np.ones(A.shape[0])
    for alg in (
        LandweberHyperplane(A_in, b),
        LandweberHalfspace(A_in, b),
        LandweberHyperslab(A_in, -b, b),
    ):
        assert np.isclose(alg.step_size, 1 / (1.01 * L), rtol=1e-4)
        assert alg.step_size < 2 / L


def test_custom_step_size(A):
    b = np.ones(A.shape[0])
    assert LandweberHyperplane(A, b, step_size=0.1).step_size == 0.1
    assert LandweberHalfspace(A, b, step_size=0.1).step_size == 0.1
    assert LandweberHyperslab(A, -b, b, step_size=0.1).step_size == 0.1


@pytest.mark.parametrize("step_size", [0.0, -1.0])
def test_invalid_step_size(A, step_size):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError):
        LandweberHyperplane(A, b, step_size=step_size)
    with pytest.raises(ValueError):
        LandweberHalfspace(A, b, step_size=step_size)
    with pytest.raises(ValueError):
        LandweberHyperslab(A, -b, b, step_size=step_size)


# ---------------------------------------------------------------------------
# single iterations
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("sparse_A", [False, True])
def test_LandweberHyperplane_project(A, x0, sparse_A):
    b = np.array([1.0, 0.0, 1.0, 3.0])
    gamma = 0.1
    alg = LandweberHyperplane(sparse.csr_array(A) if sparse_A else A, b, step_size=gamma)
    expected = x0 + gamma * A.T @ (b - A @ x0)
    assert np.allclose(alg.project(x0.copy()), expected)


@pytest.mark.parametrize("sparse_A", [False, True])
def test_LandweberHalfspace_project(A, x0, sparse_A):
    b = np.array([1.0, 0.0, 1.0, 3.0])
    gamma = 0.1
    alg = LandweberHalfspace(sparse.csr_array(A) if sparse_A else A, b, step_size=gamma)
    expected = x0 + gamma * A.T @ np.minimum(b - A @ x0, 0)
    assert np.allclose(alg.project(x0.copy()), expected)


@pytest.mark.parametrize("sparse_A", [False, True])
def test_LandweberHyperslab_project(A, x0, sparse_A):
    lb = np.array([0.0, -0.5, 0.0, 1.0])
    ub = np.array([1.0, 0.5, 1.0, 3.0])
    gamma = 0.1
    alg = LandweberHyperslab(sparse.csr_array(A) if sparse_A else A, lb, ub, step_size=gamma)
    p = A @ x0
    expected = x0 + gamma * A.T @ (np.clip(p, lb, ub) - p)
    assert np.allclose(alg.project(x0.copy()), expected)


def test_feasible_point_is_fixed(A):
    x = np.array([0.5, 0.5])
    p = A @ x
    assert np.allclose(LandweberHyperplane(A, p).project(x.copy()), x)
    assert np.allclose(LandweberHalfspace(A, p + 1).project(x.copy()), x)
    assert np.allclose(LandweberHyperslab(A, p - 1, p + 1).project(x.copy()), x)


def test_matches_reweighted_SimultaneousKaczmarz(A, x0):
    # Cimmino with w_i ~ ||a_i||^2 is Landweber with step relaxation / ||A||_F^2
    b = np.array([1.0, 0.0, 1.0, 3.0])
    row_sq = (A**2).sum(axis=1)
    fro2 = row_sq.sum()
    gamma = 0.1
    cimmino = SimultaneousKaczmarzMethod(A, b, gamma * fro2, weights=row_sq / fro2)
    landweber = LandweberHyperplane(A, b, step_size=gamma)
    assert np.allclose(cimmino.project(x0.copy()), landweber.project(x0.copy()))


# ---------------------------------------------------------------------------
# weights
# ---------------------------------------------------------------------------


@pytest.fixture
def w():
    return np.array([2.0, 0.5, 1.0, 0.25])


@pytest.mark.parametrize("sparse_A", [False, True])
def test_weighted_project(A, x0, w, sparse_A):
    A_in = sparse.csr_array(A) if sparse_A else A
    b = np.array([1.0, 0.0, 1.0, 3.0])
    lb = np.array([0.0, -0.5, 0.0, 1.0])
    gamma = 0.1
    p = A @ x0

    alg = LandweberHyperplane(A_in, b, step_size=gamma, weights=w)
    assert np.allclose(alg.project(x0.copy()), x0 + gamma * A.T @ (w * (b - p)))

    alg = LandweberHalfspace(A_in, b, step_size=gamma, weights=w)
    assert np.allclose(alg.project(x0.copy()), x0 + gamma * A.T @ (w * np.minimum(b - p, 0)))

    alg = LandweberHyperslab(A_in, lb, b, step_size=gamma, weights=w)
    assert np.allclose(alg.project(x0.copy()), x0 + gamma * A.T @ (w * (np.clip(p, lb, b) - p)))


def test_weighted_default_step_size(A, w):
    L_w = np.linalg.eigvalsh(A.T @ (w[:, None] * A)).max()
    b = np.ones(A.shape[0])
    for alg in (
        LandweberHyperplane(A, b, weights=w),
        LandweberHalfspace(A, b, weights=w),
        LandweberHyperslab(A, -b, b, weights=w),
    ):
        assert np.isclose(alg.step_size, 1 / (1.01 * L_w), rtol=1e-4)


def test_weights_used_as_given_and_normalized_for_proximity(A, w):
    b = np.ones(A.shape[0])
    alg = LandweberHyperplane(A, b, weights=w)
    assert np.allclose(alg.landweber_weights, w)
    assert np.allclose(alg.weights, w / w.sum())
    # no weights: W = I in the update, uniform weights for the proximity
    alg = LandweberHyperplane(A, b)
    assert alg.landweber_weights is None
    assert np.allclose(alg.weights, np.ones(4) / 4)


def test_scaled_weights_with_default_step_give_same_iterate(A, x0, w):
    # the default step scales with 1 / lambda_max(A^T W A), so scaling W cancels out
    lb = np.array([0.0, -0.5, 0.0, 1.0])
    ub = np.array([1.0, 0.5, 1.0, 3.0])
    x1 = LandweberHyperslab(A, lb, ub, weights=w).project(x0.copy())
    x2 = LandweberHyperslab(A, lb, ub, weights=10 * w).project(x0.copy())
    assert np.allclose(x1, x2)


@pytest.mark.parametrize(
    "bad_weights",
    [np.array([1.0, -1.0, 1.0, 1.0]), np.zeros(4), np.ones(3)],
)
def test_invalid_weights(A, bad_weights):
    b = np.ones(A.shape[0])
    with pytest.raises(ValueError):
        LandweberHyperplane(A, b, weights=bad_weights)
    with pytest.raises(ValueError):
        LandweberHalfspace(A, b, weights=bad_weights)
    with pytest.raises(ValueError):
        LandweberHyperslab(A, -b, b, weights=bad_weights)


# ---------------------------------------------------------------------------
# convergence
# ---------------------------------------------------------------------------

SOLVE_KW = {"max_iter": 5000, "prox_tol": 1e-14, "del_prox_tol": 0}


@pytest.mark.parametrize("weights", [None, np.array([2.0, 0.5, 1.0, 0.25])])
def test_LandweberHyperplane_solve(A, weights):
    x_true = np.array([0.3, -0.7])
    b = A @ x_true
    x = LandweberHyperplane(A, b, weights=weights).solve(np.zeros(2), **SOLVE_KW)
    assert np.allclose(x, x_true, atol=1e-6)


@pytest.mark.parametrize("weights", [None, np.array([2.0, 0.5, 1.0, 0.25])])
def test_LandweberHalfspace_solve(A, x0, weights):
    b = np.array([1.0, 0.0, 1.0, 3.0])
    x = LandweberHalfspace(A, b, weights=weights).solve(x0.copy(), **SOLVE_KW)
    assert np.all(A @ x <= b + 1e-6)


@pytest.mark.parametrize("weights", [None, np.array([2.0, 0.5, 1.0, 0.25])])
def test_LandweberHyperslab_solve(A, x0, weights):
    lb = np.array([0.0, -0.5, 0.0, 1.0])
    ub = np.array([1.0, 0.5, 1.0, 3.0])
    x = LandweberHyperslab(A, lb, ub, weights=weights).solve(x0.copy(), **SOLVE_KW)
    p = A @ x
    assert np.all(p >= lb - 1e-6) and np.all(p <= ub + 1e-6)
