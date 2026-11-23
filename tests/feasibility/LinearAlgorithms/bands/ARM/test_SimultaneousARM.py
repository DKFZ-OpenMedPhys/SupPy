import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SimultaneousARM
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


def test_SimultaneousARM_constructor_full(get_ARM_variables_full):
    """Test the SimultaneousARM constructor."""
    A, lb, ub = get_ARM_variables_full

    # check without given weights
    alg = SimultaneousARM(A, lb, ub)
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.weights, np.ones(len(A)) / len(A))

    alg = SimultaneousARM(A, lb, ub, weights=np.array([1 / 3, 2 / 3]))
    assert np.array_equal(alg.weights, np.array([1 / 3, 2 / 3]))
    # check with given weights
    alg = SimultaneousARM(A, lb, ub, weights=np.ones(len(A)))
    assert np.array_equal(alg.weights, np.ones(len(A)) / len(A))

    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SimultaneousARM_constructor_sparse(get_ARM_variables_sparse):
    """Test the SimultaneousARM constructor."""
    A, lb, ub = get_ARM_variables_sparse

    # check without given weights
    alg = SimultaneousARM(A, lb, ub)
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])

    alg = SimultaneousARM(A, lb, ub, weights=np.array([1 / 3, 2 / 3]))
    assert np.array_equal(alg.weights, np.array([1 / 3, 2 / 3]))
    # check with given weights
    alg = SimultaneousARM(A, lb, ub, weights=np.ones(A.shape[0]))
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])

    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SimultaneousARM_step_full(get_ARM_variables_full):
    """Test the step function of the SimultaneousARM class."""
    A, lb, ub = get_ARM_variables_full
    alg = SimultaneousARM(A, lb, ub)

    x_1 = np.array([0.0, 0.0])
    x_2 = np.array([1.0, 1.0])
    x_3 = np.array([-1.0, -1.0])
    x_4 = np.array([-1.0, 1.0])
    x_5 = np.array([0.0, 2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.0, 0.0])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - 13 / 16 * np.array([1.0, 1.0])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - 13 / 16 * np.array([-1.0, -1.0])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - 13 / 16 * np.array([-1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0, 13 / 8]))) < 1e-10


def test_SimultaneousARM_step_sparse(get_ARM_variables_sparse):
    """Test the step function of the SimultaneousARM class."""
    A, lb, ub = get_ARM_variables_sparse

    alg = SimultaneousARM(A, lb, ub, weights=np.ones(A.shape[0]))

    x_1 = np.array([0.0, 0.0])
    x_2 = np.array([1.0, 1.0])
    x_3 = np.array([-1.0, -1.0])
    x_4 = np.array([-1.0, 1.0])
    x_5 = np.array([0.0, 2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.0, 0.0])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - 13 / 16 * np.array([1.0, 1.0])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - 13 / 16 * np.array([-1.0, -1.0])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - 13 / 16 * np.array([-1.0, 1.0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0, 13 / 8])) < 1e-10)


def test_SimultaneousARM_proximity():
    """Test the proximity function of the SimultaneousARM class."""
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    alg = SimultaneousARM(A, lb, ub, weights=np.ones(len(A)) / len(A))

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

    assert np.abs(prox_1[0] - 0.04) < 1e-10
    assert np.abs(prox_1[1] - 0.4) < 1e-10

    assert np.abs(prox_2[0] - 9 / 8) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 0.04) < 1e-10
    assert np.abs(prox_3[1] - 0.4) < 1e-10

    assert np.abs(prox_4[0] - 9 / 8) < 1e-10
    assert np.abs(prox_4[1] - 2) < 1e-10

    assert np.abs(prox_5[0] - 9 / 8) < 1e-10
    assert np.abs(prox_5[1] - 2) < 1e-10


def test_SimultaneousARM_proximity_weighted():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    alg = SimultaneousARM(A, lb, ub, weights=np.array([1 / 2, 1 / 4, 1 / 8, 1 / 8]))

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

    assert np.abs(prox_1[0] - 0.08) < 1e-10
    assert np.abs(prox_1[1] - 0.4) < 1e-10
    assert np.abs(prox_2[0] - 33 / 16) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 0.08) < 1e-10
    assert np.abs(prox_3[1] - 0.4) < 1e-10

    assert np.abs(prox_4[0] - 33 / 16) < 1e-10
    assert np.abs(prox_4[1] - 2) < 1e-10

    assert np.abs(prox_5[0] - 17 / 16) < 1e-10
    assert np.abs(prox_5[1] - 2) < 1e-10


# ---------------------------------------------------------------------------
# SimultaneousARM: options and branches
# ---------------------------------------------------------------------------


def test_SimultaneousARM_plain_list_weights(get_ARM_variables_full):
    # plain lists are allowed per the type hint; [0.5, 0.5] equals the
    # default uniform weights
    A, lb, ub = get_ARM_variables_full
    alg_list = SimultaneousARM(A, lb, ub, weights=[0.5, 0.5])
    alg_default = SimultaneousARM(A, lb, ub)
    assert np.allclose(alg_list.weights, alg_default.weights)

    x = np.array([1.0, 1.0])
    assert np.allclose(alg_list.project(x.copy()), alg_default.project(x.copy()))


def test_SimultaneousARM_plain_list_weights_renormalized(get_ARM_variables_full):
    # a plain list that does not sum to 1 is renormalized with a warning
    # (used to crash on list.sum() before weights were coerced to an array)
    A, lb, ub = get_ARM_variables_full
    with pytest.warns(UserWarning, match="Renormalizing"):
        alg = SimultaneousARM(A, lb, ub, weights=[1.0, 1.0])
    assert np.allclose(alg.weights, np.array([0.5, 0.5]))


def test_SimultaneousARM_weighted_proximity(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SimultaneousARM(A, lb, ub, weights=np.array([0.75, 0.25]))
    # x = [1, 1]: band 0 exceeded by 1, band 1 satisfied:
    # p_norm 2 -> 0.75 * 1^2 = 0.75; max_norm -> 1
    prox = alg.proximity(np.array([1.0, 1.0]), [("p_norm", 2), "max_norm"])
    assert np.abs(prox[0] - 0.75) < 1e-10
    assert np.abs(prox[1] - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_SimultaneousARM_fejer_monotone(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SimultaneousARM(A, lb, ub)
    z = np.zeros(2)  # interior point of the feasible set
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_SimultaneousARM_project_idempotent_on_feasible(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SimultaneousARM(A, lb, ub)
    x = np.array([0.0, 0.0])
    assert np.array_equal(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SimultaneousARM_solve_int_input(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = SimultaneousARM(A, lb, ub)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([1, 1]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_SimultaneousARM_cupy_backend(get_ARM_variables_full):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_ARM_variables_full
    x_np = SimultaneousARM(A, lb, ub).step(np.array([1.0, 1.0]))

    alg_cp = SimultaneousARM(cp.asarray(A), cp.asarray(lb), cp.asarray(ub))
    x_cp = alg_cp.step(cp.asarray([1.0, 1.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
