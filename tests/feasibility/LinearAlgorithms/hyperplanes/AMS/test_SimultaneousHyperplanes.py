import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SimultaneousKaczmarzMethod
from suppy.utils import LinearMapping


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


def test_SimultaneousKaczmarzMethod_no_relaxation_no_weights_constructor_full(get_full_variables):
    """
    Test the SimultaneousKaczmarzMethod constructor with no relaxation and a
    full
    matrix.
    """
    A, b = get_full_variables
    alg = SimultaneousKaczmarzMethod(A, b)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(len(A)) / len(A))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def test_SimultaneousKaczmarzMethod_no_relaxation_no_weights_constructor_sparse(
    get_sparse_variables,
):
    """
    Test the SimultaneousKaczmarzMethod constructor with no relaxation and a
    sparse
    matrix.
    """
    A, b = get_sparse_variables
    alg = SimultaneousKaczmarzMethod(A, b)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0


def testSimultaneousKaczmarzMethod_relaxation_weights_constructor_full(get_full_variables):
    """
    Test the SimultaneousKaczmarzMethod constructor with relaxation and
    custom
    weights.
    """
    A, b = get_full_variables
    alg = SimultaneousKaczmarzMethod(
        A, b, algorithmic_relaxation=1.5, relaxation=1.5, weights=np.ones(len(A))
    )

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(len(A)) / A.shape[0])
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.5


def testSimultaneousKaczmarzMethod_relaxation_weights_constructor_sparse(get_sparse_variables):
    """
    Test the SimultaneousKaczmarzMethod constructor with relaxation and
    custom
    weights.
    """
    A, b = get_sparse_variables
    alg = SimultaneousKaczmarzMethod(
        A,
        b,
        algorithmic_relaxation=1.5,
        relaxation=1.5,
        weights=np.ones(A.shape[0]),
    )

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.weights, np.ones(A.shape[0]) / A.shape[0])
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.5


def test_SimultaneousKaczmarzMethod_step_full(get_full_variables):
    """Test the step function of the SimultaneousKaczmarzMethod class."""
    A, b = get_full_variables
    alg = SimultaneousKaczmarzMethod(A, b, weights=np.ones(len(A)) / A.shape[0])

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([5 / 3, 5 / 3])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([5 / 3, 5 / 3])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([4 / 3, 2])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([5 / 6, 11 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1 / 3, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-7 / 6, -1 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SimultaneousKaczmarzMethod_step_sparse(get_sparse_variables):
    """Test the step function of the SimultaneousKaczmarzMethod class."""
    A, b = get_sparse_variables

    alg = SimultaneousKaczmarzMethod(A, b, weights=np.ones(A.shape[0]))

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([5 / 3, 5 / 3])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([5 / 3, 5 / 3])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([4 / 3, 2])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([5 / 6, 11 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1 / 3, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-7 / 6, -1 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SimultaneousKaczmarzMethod_algorithmic_relaxation_step(get_sparse_variables):
    """Test the step function with relaxation of the SimultaneousKaczmarzMethod
    class.
    """
    A, b = get_sparse_variables

    alg = SimultaneousKaczmarzMethod(A, b, weights=np.ones(A.shape[0]), algorithmic_relaxation=2.0)

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([4 / 3, 4 / 3])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([4 / 3, 4 / 3])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([5 / 3, 1])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([10 / 6, 4 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([4 / 3, 0])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([4 / 6, -2 / 6])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SimultaneousKaczmarzMethod_relaxation_step(get_sparse_variables):
    """Test the step function with relaxation of the SimultaneousKaczmarzMethod
    class.
    """
    A, b = get_sparse_variables

    alg = SimultaneousKaczmarzMethod(A, b, weights=np.ones(A.shape[0]), relaxation=2.0)
    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    # check that project gives the same result as step
    x_1 = np.array([2.0, 2.0])
    x_n = alg.project(x_1)
    assert np.all(abs(x_n - np.array([4 / 3, 4 / 3])) < 1e-10)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([5 / 3, 1])) < 1e-10)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([10 / 6, 4 / 6])) < 1e-10)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([4 / 3, 0])) < 1e-10)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([4 / 6, -2 / 6])) < 1e-10)


def test_SimultaneousKaczmarzMethod_proximity(get_full_variables):
    """Test the proximity function of the SimultaneousKaczmarzMethod class."""
    A, b = get_full_variables
    alg = SimultaneousKaczmarzMethod(A, b, weights=np.ones(len(A)) / len(A))

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    prox_measures = []
    no_prox = alg.proximity(x_1, prox_measures)
    assert no_prox.size == 0

    prox_measures = [("p_norm", 2), "max_norm"]
    prox_1 = alg.proximity(x_1, prox_measures)
    prox_2 = alg.proximity(x_2, prox_measures)
    prox_3 = alg.proximity(x_3, prox_measures)
    prox_4 = alg.proximity(x_4, prox_measures)
    prox_5 = alg.proximity(x_5, prox_measures)

    assert np.abs(prox_1[0] - 2 / 3) < 1e-10
    assert np.abs(prox_1[1] - 1) < 1e-10

    assert np.abs(prox_2[0] - 5 / 3) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 29 / 12) < 1e-10
    assert np.abs(prox_3[1] - 2) < 1e-10

    assert np.abs(prox_4[0] - 14 / 3) < 1e-10
    assert np.abs(prox_4[1] - 3) < 1e-10

    assert np.abs(prox_5[0] - 77 / 12) < 1e-10
    assert np.abs(prox_5[1] - 4) < 1e-10


# ---------------------------------------------------------------------------
# SimultaneousKaczmarzMethod: options and branches
# ---------------------------------------------------------------------------


def test_SimultaneousKaczmarzMethod_plain_list_weights(get_full_variables):
    # plain lists are allowed per the type hint; weights summing to 1 are
    # stored unchanged
    A, b = get_full_variables
    alg = SimultaneousKaczmarzMethod(A, b, weights=[0.5, 0.25, 0.25])
    assert np.allclose(alg.weights, np.array([0.5, 0.25, 0.25]))

    # x = [2, 2]: residuals b - Ax = [-1, 0, -1], row norms^2 = [1, 0.5, 1]:
    # x += 0.5*(-1)*[1, 0] + 0.25*(-1)*[0, 1] = [1.5, 1.75]
    x_n = alg.step(np.array([2.0, 2.0]))
    assert np.allclose(x_n, np.array([1.5, 1.75]))


def test_SimultaneousKaczmarzMethod_plain_list_weights_renormalized(get_full_variables):
    # a plain list that does not sum to 1 is renormalized with a warning
    # (used to crash on list.sum() before weights were coerced to an array)
    A, b = get_full_variables
    with pytest.warns(UserWarning, match="Renormalizing"):
        alg = SimultaneousKaczmarzMethod(A, b, weights=[1.0, 1.0, 1.0])
    assert np.allclose(alg.weights, np.ones(3) / 3)

    alg_default = SimultaneousKaczmarzMethod(A, b)
    x = np.array([2.0, 2.0])
    assert np.allclose(alg.project(x.copy()), alg_default.project(x.copy()))


def test_SimultaneousKaczmarzMethod_proximity_flag_false(get_full_variables):
    A, b = get_full_variables
    alg = SimultaneousKaczmarzMethod(A, b, proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_SimultaneousKaczmarzMethod_solve_converges_to_solution(get_full_variables):
    # the system is consistent with unique solution [1, 1]
    A, b = get_full_variables
    rng = np.random.default_rng(42)
    for _ in range(3):
        alg = SimultaneousKaczmarzMethod(A, b)
        x = 4 * rng.standard_normal(2)
        x_n = alg.solve(x, max_iter=1000)
        assert np.allclose(x_n, np.array([1.0, 1.0]), atol=1e-2)
        assert alg.proximities[-1][0] < alg.proximities[0][0]


def test_SimultaneousKaczmarzMethod_project_idempotent_on_solution(get_full_variables):
    A, b = get_full_variables
    alg = SimultaneousKaczmarzMethod(A, b)
    x = np.array([1.0, 1.0])  # the solution point
    assert np.allclose(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SimultaneousKaczmarzMethod_solve_int_input(get_full_variables):
    A, b = get_full_variables
    alg = SimultaneousKaczmarzMethod(A, b)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_SimultaneousKaczmarzMethod_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    x_np = SimultaneousKaczmarzMethod(A, b).step(np.array([2.0, 2.0]))

    alg_cp = SimultaneousKaczmarzMethod(cp.asarray(A), cp.asarray(b))
    x_cp = alg_cp.step(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
