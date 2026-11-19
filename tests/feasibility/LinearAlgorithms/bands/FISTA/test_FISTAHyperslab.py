import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import FISTAHyperslab
from suppy.projections import BoxProjection, SequentialProjection
from suppy.perturbations import PowerSeriesGradientPerturbation
from suppy.superiorization import Superiorization


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


def test_FISTAHyperslab_lipschitz_estimate(get_full_variables):
    # equal weights 1/4: A^T W A = 3/4 I
    A, lb, ub = get_full_variables
    alg = FISTAHyperslab(A, lb, ub)
    assert alg.lipschitz == pytest.approx(1.01 * 0.75)


def test_FISTAHyperslab_lipschitz_estimate_weighted():
    rng = np.random.default_rng(1)
    A = rng.random((20, 5))
    w = rng.random(20)
    w /= w.sum()
    alg = FISTAHyperslab(A, np.zeros(20), np.ones(20), weights=w)
    lam_max = np.linalg.eigvalsh(A.T @ (w[:, None] * A)).max()
    assert alg.lipschitz == pytest.approx(1.01 * lam_max, rel=1e-5)


def test_FISTAHyperslab_lipschitz_given(get_full_variables):
    A, lb, ub = get_full_variables
    alg = FISTAHyperslab(A, lb, ub, lipschitz=2.0)
    assert alg.lipschitz == 2.0


def test_FISTAHyperslab_lipschitz_zero_matrix():
    with pytest.raises(ValueError), np.errstate(divide="ignore"):
        FISTAHyperslab(np.zeros((2, 2)), np.zeros(2), np.ones(2))


def test_FISTAHyperslab_project(get_full_variables):
    # violations (Ax - ub) of rows 0, 2, 3: 2, 0.5, 0.5 -> gradient 1/4 * (2.5, 2.5)
    A, lb, ub = get_full_variables
    alg = FISTAHyperslab(A, lb, ub)
    proj = alg.project(np.array([2.0, 2.0]))
    assert np.allclose(proj, 2.0 - 0.625 / alg.lipschitz)

    alg.reset()
    proj = alg.project(np.array([0.0, 0.0]))
    assert np.array_equal(proj, np.array([0.0, 0.0]))


def test_FISTAHyperslab_project_sparse(get_full_variables, get_sparse_variables):
    alg_full = FISTAHyperslab(*get_full_variables)
    alg_sparse = FISTAHyperslab(*get_sparse_variables)
    assert alg_sparse.lipschitz == pytest.approx(alg_full.lipschitz)
    for x in [np.array([2.0, 2.0]), np.array([1.0, -3.0]), np.array([0.5, 4.0])]:
        assert np.allclose(alg_sparse.project(x), alg_full.project(x))


def test_FISTAHyperslab_momentum():
    # f(x) = x^2 / 2, step 1/L = 1/2: output = z / 2 with z the momentum point
    alg = FISTAHyperslab(np.array([[1.0]]), np.array([0.0]), np.array([0.0]), lipschitz=2.0)
    assert alg.project(np.array([4.0]))[0] == pytest.approx(2.0)  # z = x
    assert alg.project(np.array([2.0]))[0] == pytest.approx(1.0)  # factor (t_1 - 1) / t_2 = 0
    # Beck-Teboulle: t_2 = (1 + sqrt(5)) / 2, t_3 = (1 + sqrt(1 + 4 t_2^2)) / 2
    t2 = (1 + np.sqrt(5)) / 2
    t3 = (1 + np.sqrt(1 + 4 * t2**2)) / 2
    z = 1 + (t2 - 1) / t3 * (1 - 2)
    assert alg.project(np.array([1.0]))[0] == pytest.approx(z / 2)

    alg.reset()
    assert alg.project(np.array([1.0]))[0] == pytest.approx(0.5)


def test_FISTAHyperslab_project_keeps_dtype(get_full_variables):
    A, lb, ub = get_full_variables
    alg = FISTAHyperslab(A.astype(np.float32), lb.astype(np.float32), ub.astype(np.float32))
    assert alg.project(np.array([2.0, 2.0], dtype=np.float32)).dtype == np.float32


def test_FISTAHyperslab_project_does_not_modify_input(get_full_variables):
    alg = FISTAHyperslab(*get_full_variables)
    x = np.array([2.0, 2.0])
    alg.project(x)
    assert np.array_equal(x, np.array([2.0, 2.0]))


def test_FISTAHyperslab_solve(get_full_variables):
    A, lb, ub = get_full_variables
    alg = FISTAHyperslab(A, lb, ub)
    x = alg.solve(np.array([5.0, 5.0]), max_iter=1000)
    assert alg.proximity(x, [("p_norm", 2)])[0] < 1e-6

    # solve resets the momentum, a second call gives the same result
    x2 = alg.solve(np.array([5.0, 5.0]), max_iter=1000)
    assert np.array_equal(x, x2)


def test_FISTAHyperslab_reset_in_sequential_projection(get_full_variables):
    alg = FISTAHyperslab(*get_full_variables)
    seq = SequentialProjection([alg, BoxProjection(0, np.inf)])
    seq.project(np.array([2.0, 2.0]))
    assert alg._x_prev is not None
    seq.reset()
    assert alg._x_prev is None and alg._k == 0 and alg._t == 1.0


def test_FISTAHyperslab_superiorization_resets(get_full_variables):
    alg = FISTAHyperslab(*get_full_variables)
    pert = PowerSeriesGradientPerturbation(lambda x: (x - 3) @ (x - 3), lambda x: 2 * (x - 3))
    sup = Superiorization(alg, pert)
    x1 = sup.solve(np.array([5.0, 5.0]), max_iter=50)
    x2 = sup.solve(np.array([5.0, 5.0]), max_iter=50)
    assert np.array_equal(x1, x2)
