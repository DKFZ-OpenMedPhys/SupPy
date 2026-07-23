import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialARM
from suppy.utils import LinearMapping


@pytest.fixture
def get_ARM_variables_full():
    A = np.array([[1, 1], [-1, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    return A, lb, ub


@pytest.fixture
def get_ARM_variables_sparse():
    A = sparse.csr_matrix([[1, 1], [-1, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    return A, lb, ub


def test_SequentialARM_constructor_full(get_ARM_variables_full):
    """Test the SequentialARM constructor."""
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SequentialARM_constructor_sparse(get_ARM_variables_sparse):
    """Test the SequentialARM constructor."""
    A, lb, ub = get_ARM_variables_sparse
    alg = SequentialARM(A, lb, ub)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SequentialARM_step_full(get_ARM_variables_full):
    """Test the step function of the SequentialARM class."""
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub)

    x_1 = np.array([0.0, 0.0])
    x_2 = np.array([1.0, 1.0])
    x_3 = np.array([-1.0, -1.0])
    x_4 = np.array([-1.0, 1.0])
    x_5 = np.array([0.0, 2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.0, 0.0])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - 5 / 8 * np.array([1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - 5 / 8 * np.array([-1.0, -1.0])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - 5 / 8 * np.array([-1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0, 5 / 4])) < 1e-10)


def test_SequentialARM_step_sparse(get_ARM_variables_sparse):
    """Test the step function of the SequentialARM class."""
    A, lb, ub = get_ARM_variables_sparse
    alg = SequentialARM(A, lb, ub)

    x_1 = np.array([0.0, 0.0])
    x_2 = np.array([1.0, 1.0])
    x_3 = np.array([-1.0, -1.0])
    x_4 = np.array([-1.0, 1.0])
    x_5 = np.array([0.0, 2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.0, 0.0])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - 5 / 8 * np.array([1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - 5 / 8 * np.array([-1.0, -1.0])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - 5 / 8 * np.array([-1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0, 5 / 4])) < 1e-10)


def test_SequentialARM_solve(get_ARM_variables_full):
    """Test the solve function of the SequentialARM class."""
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub)
    x_0 = np.array([1.0, 1.0])
    x = alg.solve(x_0, prox_tol=1e-6)
    assert alg.proximities[-1][0] < 1e-6

    x_0 = np.array([1.0, 1.0])
    x = alg.solve(x_0, prox_tol=1e-6, proximity_measures=["max_norm"])
    assert np.all(abs(x - np.array([0.5, 0.5])) < 1e-6)

    x_0 = np.array([1.0, 1.0])
    x = alg.solve(x_0, max_iter=2)
    assert np.all(abs(x - np.array([41 / 80, 41 / 80])) < 1e-10)

    x_0 = np.array([1.0, 1.0])
    x = alg.solve(x_0, max_iter=2, proximity_measures=[("p_norm", 2), "max_norm"])
    assert np.all(abs(x - np.array([41 / 80, 41 / 80])) < 1e-10)


# ---------------------------------------------------------------------------
# SequentialARM: options and branches
# ---------------------------------------------------------------------------


def test_SequentialARM_custom_cs_subset(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    # only constraint 0 is visited: x = [2, 0] -> d = 2, psi = 1,
    # step = -1/2 * 1/2 * (4 - 1)/2 = -0.375 along [1, 1]
    alg = SequentialARM(A, lb, ub, cs=[0])
    x_n = alg.step(np.array([2.0, 0.0]))
    assert np.all(abs(x_n - np.array([1.625, -0.375])) < 1e-10)


def test_SequentialARM_algorithmic_relaxation(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub, algorithmic_relaxation=2.0)
    # x = [1, 1]: d = 2 on row 0, step doubles to -0.75 along [1, 1];
    # row 1 stays satisfied
    x_n = alg.step(np.array([1.0, 1.0]))
    assert np.all(abs(x_n - np.array([0.25, 0.25])) < 1e-10)


def test_SequentialARM_proximity_flag_false(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub, proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_SequentialARM_fejer_monotone(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub)
    z = np.zeros(2)  # interior point of the feasible set
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_SequentialARM_project_idempotent_on_feasible(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub)
    x = np.array([0.0, 0.0])
    assert np.array_equal(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SequentialARM_solve_int_input(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SequentialARM(A, lb, ub)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([1, 1]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_SequentialARM_cupy_backend(get_ARM_variables_full):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_ARM_variables_full
    x_np = SequentialARM(A, lb, ub).step(np.array([1.0, 1.0]))

    alg_cp = SequentialARM(cp.asarray(A), cp.asarray(lb), cp.asarray(ub))
    x_cp = alg_cp.step(cp.asarray([1.0, 1.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
