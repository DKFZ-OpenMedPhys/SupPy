import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialWeightedKaczmarz
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


@pytest.fixture
def get_SequentialWeightedKaczmarz_input(get_full_variables):
    A, b = get_full_variables
    return SequentialWeightedKaczmarz(A, b, algorithmic_relaxation=1.5), A, b


@pytest.fixture
def get_SequentialWeightedKaczmarz_input_sparse(get_sparse_variables):
    A, b = get_sparse_variables
    return SequentialWeightedKaczmarz(A, b, algorithmic_relaxation=1.5), A, b


def test_SequentialWeightedKaczmarz_constructor_no_weights_full(
    get_SequentialWeightedKaczmarz_input,
):
    """
    Test the SequentialWeightedKaczmarz constructor with no weights and
    full
    matrix.
    """
    alg, A, b = get_SequentialWeightedKaczmarz_input

    alg = SequentialWeightedKaczmarz(A, b, algorithmic_relaxation=1.5)
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(len(A)))


def test_SequentialWeightedKaczmarz_constructor_no_weights_sparse(
    get_SequentialWeightedKaczmarz_input_sparse,
):
    """
    Test the SequentialWeightedKaczmarz constructor with no weights and
    sparse
    matrix.
    """
    alg, A, b = get_SequentialWeightedKaczmarz_input_sparse

    alg = SequentialWeightedKaczmarz(A, b, algorithmic_relaxation=1.5)

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(A.shape[0]))


def test_SequentialWeightedKaczmarz_constructor_custom_weights(get_full_variables):
    """Test the SequentialWeightedKaczmarz constructor with custom
    weights.
    """
    A, b = get_full_variables
    alg = SequentialWeightedKaczmarz(A, b, algorithmic_relaxation=1.5, weights=np.ones(len(A)))

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0
    assert np.all(alg.weights == np.ones(len(A)))
    assert alg.temp_weight_decay == 1.0

    # does weight decay work?


def test_SequentialWeightedKaczmarz_weight_decay_full(get_full_variables):
    """Test the weight decay of the SequentialWeightedKaczmarz class."""
    A, b = get_full_variables
    alg = SequentialWeightedKaczmarz(A, b, weights=np.ones(len(A)), weight_decay=0.5)

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)
    assert alg.temp_weight_decay == 0.5


def test_SequentialWeightedKaczmarz_weight_decay_sparse(get_sparse_variables):
    """Test the weight decay of the SequentialWeightedKaczmarz class."""
    A, b = get_sparse_variables
    alg = SequentialWeightedKaczmarz(A, b, weights=np.ones(A.shape[0]), weight_decay=0.5)

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)
    assert alg.temp_weight_decay == 0.5


def test_SequentialWeightedKaczmarz_weight_decay_step_full(get_full_variables):
    """Test the weight decay of the SequentialWeightedKaczmarz class."""
    A, b = get_full_variables
    alg = SequentialWeightedKaczmarz(A, b, algorithmic_relaxation=1.5, weights=np.ones(len(A)))
    alg.temp_weight_decay = 2 / 3
    assert alg.temp_weight_decay == 2 / 3

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([2, 1])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([2, 1])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SequentialWeightedKaczmarz_weight_decay_step_sparse(get_sparse_variables):
    """Test the weight decay of the SequentialWeightedKaczmarz class."""
    A, b = get_sparse_variables
    alg = SequentialWeightedKaczmarz(A, b, algorithmic_relaxation=1.5, weights=np.ones(A.shape[0]))
    alg.temp_weight_decay = 2 / 3
    assert alg.temp_weight_decay == 2 / 3

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([2, 1])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([2, 1])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([1.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([0.5, 1])) < 1e-10)
    assert np.array_equal(x_n, x_5)


# ---------------------------------------------------------------------------
# SequentialWeightedKaczmarz: options and branches
# ---------------------------------------------------------------------------


def test_SequentialWeightedKaczmarz_weight_decay_accumulates(get_full_variables):
    A, b = get_full_variables
    alg = SequentialWeightedKaczmarz(A, b, weight_decay=0.5)
    assert alg.temp_weight_decay == 1.0
    # the decay is applied multiplicatively after every projection sweep
    alg.step(np.array([2.0, 2.0]))
    assert alg.temp_weight_decay == 0.5
    alg.step(np.array([2.0, 2.0]))
    assert alg.temp_weight_decay == 0.25


def test_SequentialWeightedKaczmarz_proximity_flag_false(get_full_variables):
    A, b = get_full_variables
    alg = SequentialWeightedKaczmarz(A, b, proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_SequentialWeightedKaczmarz_defaults_match_sequential(get_full_variables):
    # with unit weights and no decay every update reduces to the plain
    # Kaczmarz sweep
    from suppy.feasibility import KaczmarzMethod

    A, b = get_full_variables
    alg_w = SequentialWeightedKaczmarz(A, b)
    alg_s = KaczmarzMethod(A, b)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_w.project(x.copy()), alg_s.project(x.copy()))


def test_SequentialWeightedKaczmarz_project_idempotent_on_solution(get_full_variables):
    A, b = get_full_variables
    alg = SequentialWeightedKaczmarz(A, b)
    x = np.array([1.0, 1.0])  # the solution point
    assert np.allclose(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SequentialWeightedKaczmarz_solve_int_input(get_full_variables):
    A, b = get_full_variables
    alg = SequentialWeightedKaczmarz(A, b)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_SequentialWeightedKaczmarz_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    x_np = SequentialWeightedKaczmarz(A, b).step(np.array([2.0, 2.0]))

    alg_cp = SequentialWeightedKaczmarz(cp.asarray(A), cp.asarray(b))
    x_cp = alg_cp.step(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
