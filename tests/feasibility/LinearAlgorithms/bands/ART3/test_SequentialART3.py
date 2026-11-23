import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialART3plus
from suppy.utils import LinearMapping


@pytest.fixture
def get_ART3_variables_full():
    A = np.array([[1, 0], [0, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    return A, lb, ub


@pytest.fixture
def get_ART3_variables_sparse():
    A = sparse.csr_matrix([[1, 0], [0, 1]])
    lb = np.array([-1, -1])
    ub = np.array([1, 1])
    return A, lb, ub


def test_SequentialART3plus_constructor_full(get_ART3_variables_full):
    """Test the SequentialART3plus constructor."""
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub, cs=[1, 0])
    assert alg.cs == [1, 0]

    alg = SequentialART3plus(A, lb, ub)
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(len(A)))


def test_SequentialART3plus_constructor_sparse(get_ART3_variables_sparse):
    """Test the SequentialART3plus constructor."""
    A, lb, ub = get_ART3_variables_sparse
    alg = SequentialART3plus(A, lb, ub)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))


def test_SequentialART3plus_step_full(get_ART3_variables_full):
    """Test the step function of the SequentialART3plus class."""
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub)

    x_1 = np.array([-3.0, 0.0])
    x_2 = np.array([-1.5, 0.0])
    x_3 = np.array([0.5, 0.5])
    x_4 = np.array([1.5, 0.0])
    x_5 = np.array([3.0, 0.0])
    x_6 = np.array([-3.0, -3.0])
    x_7 = np.array([-3.0, 3.0])
    x_8 = np.array([-3, 1.5])

    x_n = alg.step(x_1)
    assert np.array_equal(x_1, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_2)
    assert np.array_equal(x_2, x_n)
    assert np.array_equal(x_n, np.array([-0.5, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_3)
    assert np.array_equal(x_3, x_n)
    assert np.array_equal(x_n, np.array([0.5, 0.5]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_4)
    assert np.array_equal(x_4, x_n)
    assert np.array_equal(x_n, np.array([0.5, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_5)
    assert np.array_equal(x_5, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_6)
    assert np.array_equal(x_6, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_7)
    assert np.array_equal(x_7, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_8)
    assert np.array_equal(x_8, x_n)
    assert np.array_equal(x_n, np.array([0, 0.5]))


def test_SequentialART3plus_step_sparse(get_ART3_variables_sparse):
    """Test the step function of the SequentialART3plus class."""
    A, lb, ub = get_ART3_variables_sparse
    alg = SequentialART3plus(A, lb, ub)

    x_1 = np.array([-3.0, 0.0])
    x_2 = np.array([-1.5, 0.0])
    x_3 = np.array([0.5, 0.5])
    x_4 = np.array([1.5, 0.0])
    x_5 = np.array([3.0, 0.0])
    x_6 = np.array([-3.0, -3.0])
    x_7 = np.array([-3.0, 3.0])
    x_8 = np.array([-3, 1.5])

    x_n = alg.step(x_1)
    assert np.array_equal(x_1, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_2)
    assert np.array_equal(x_2, x_n)
    assert np.array_equal(x_n, np.array([-0.5, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_3)
    assert np.array_equal(x_3, x_n)
    assert np.array_equal(x_n, np.array([0.5, 0.5]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_4)
    assert np.array_equal(x_4, x_n)
    assert np.array_equal(x_n, np.array([0.5, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_5)
    assert np.array_equal(x_5, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_6)
    assert np.array_equal(x_6, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_7)
    assert np.array_equal(x_7, x_n)
    assert np.array_equal(x_n, np.array([0, 0]))

    alg = SequentialART3plus(A, lb, ub)
    x_n = alg.step(x_8)
    assert np.array_equal(x_8, x_n)
    assert np.array_equal(x_n, np.array([0, 0.5]))


def test_SequentialART3plus_solve(get_ART3_variables_full):
    _, lb, ub = get_ART3_variables_full
    A = np.array([[1, 0], [-1, 1]])
    alg = SequentialART3plus(A, lb, ub)
    x_0 = np.array([0, 0])
    x = alg.solve(x_0)
    assert np.array_equal(x, np.array([0, 0]))
    x_1 = np.array([10.0, 20.0])
    x = alg.solve(x_1, storage=True)
    assert np.all(abs(x - np.array([0.75, 1.25])) <= 1e-10)


# ---------------------------------------------------------------------------
# SequentialART3plus: options and branches
# ---------------------------------------------------------------------------


def test_SequentialART3plus_initial_cs_copy_semantics(get_ART3_variables_full):
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub, cs=[1, 0])
    # a feasible point prunes every constraint from the working sequence...
    alg.step(np.array([0.5, 0.5]))
    assert alg.cs == []
    # ...but the initial control sequence stays untouched
    assert alg.initial_cs == [1, 0]


def test_SequentialART3plus_partial_constraint_pruning(get_ART3_variables_full):
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub)
    # x = [3, 0.5]: constraint 0 is far outside (|p| > u - l) -> projected
    # onto the band center and kept; constraint 1 is met -> pruned
    x_n = alg.step(np.array([3.0, 0.5]))
    assert np.array_equal(x_n, np.array([0.0, 0.5]))
    assert alg.cs == [0]


def test_SequentialART3plus_solve_resets_cs(get_ART3_variables_full):
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub)
    alg.step(np.array([0.5, 0.5]))  # prunes every constraint
    assert alg.cs == []

    # solve() restores the control sequence from initial_cs before iterating
    x_n = alg.solve(np.array([3.0, 0.0]), max_iter=50)
    p = alg.map(x_n)
    assert np.all(p >= lb - 1e-6)
    assert np.all(p <= ub + 1e-6)


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_SequentialART3plus_project_idempotent_on_feasible(get_ART3_variables_full):
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub)
    x = np.array([0.5, -0.5])
    assert np.array_equal(alg.project(x.copy()), x)


def test_SequentialART3plus_solve_final_feasibility(get_ART3_variables_full):
    A, lb, ub = get_ART3_variables_full
    rng = np.random.default_rng(42)
    for _ in range(5):
        alg = SequentialART3plus(A, lb, ub)
        x = 4 * rng.standard_normal(2)
        x_n = alg.solve(x, max_iter=500)
        p = alg.map(x_n)
        assert np.all(p >= lb - 1e-6)
        assert np.all(p <= ub + 1e-6)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SequentialART3plus_solve_int_input(get_ART3_variables_full):
    A, lb, ub = get_ART3_variables_full
    alg = SequentialART3plus(A, lb, ub)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([3, 0]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_SequentialART3plus_cupy_backend(get_ART3_variables_full):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_ART3_variables_full
    x_np = SequentialART3plus(A, lb, ub).step(np.array([3.0, 0.5]))

    alg_cp = SequentialART3plus(cp.asarray(A), cp.asarray(lb), cp.asarray(ub))
    x_cp = alg_cp.step(cp.asarray([3.0, 0.5]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
