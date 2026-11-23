import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import StringAveragedKaczmarz
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


def test_StringAveragedKaczmarz_constructor_full(get_full_variables):
    """Test the StringAveragedKaczmarz constructor."""
    A, b = get_full_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0, 1, 2]])  # sequential like

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert alg.strings == [[0, 1, 2]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedKaczmarz(A, b, strings=[[0], [1], [2]])  # simultaneous like
    assert alg.strings == [[0], [1], [2]]
    assert np.array_equal(alg.weights, np.ones(3) / 3)


def test_StringAveragedKaczmarz_constructor_sparse(get_sparse_variables):
    """Test the StringAveragedKaczmarz constructor."""
    A, b = get_sparse_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0, 1, 2]])  # sequential like

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert alg.strings == [[0, 1, 2]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedKaczmarz(A, b, strings=[[0], [1], [2]])  # simultaneous like
    assert alg.strings == [[0], [1], [2]]
    assert np.array_equal(alg.weights, np.ones(3) / 3)


def test_StringAveragedKaczmarz_step_full_sequential_like(get_full_variables):
    """
    Test the step function of the StringAveragedKaczmarz class for
    sequential
    like strings.
    """

    A, b = get_full_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0, 1, 2]])  # sequential like

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


def test_StringAveragedKaczmarz_step_sparse_sequential_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedKaczmarz class for
    sequential
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0, 1, 2]])  # sequential like

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


def test_StringAveragedKaczmarz_step_full_simultaneous_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedKaczmarz class for
    simultaneous
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0], [1], [2]])  # simultaneous like

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


def test_StringAveragedKaczmarz_step_sparse_simultaneous_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedKaczmarz class for
    simultaneous
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0], [1], [2]])  # simultaneous like

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


# ---------------------------------------------------------------------------
# StringAveragedKaczmarz: options and branches
# ---------------------------------------------------------------------------


def test_StringAveragedKaczmarz_weights_length_mismatch(get_full_variables):
    A, b = get_full_variables
    with pytest.raises(ValueError):
        StringAveragedKaczmarz(A, b, strings=[[0, 1], [2]], weights=[1.0])


def test_StringAveragedKaczmarz_custom_string_weights(get_full_variables):
    A, b = get_full_variables
    # strings [0] and [2] with weights 0.7/0.3 on x = [2, 2]:
    # string [0]: project onto x = 1 -> [1, 2];
    # string [2]: project onto y = 1 -> [2, 1];
    # average: 0.7*[1, 2] + 0.3*[2, 1] = [1.3, 1.7]
    alg = StringAveragedKaczmarz(A, b, strings=[[0], [2]], weights=np.array([0.7, 0.3]))
    x_n = alg.step(np.array([2.0, 2.0]))
    assert np.all(abs(x_n - np.array([1.3, 1.7])) < 1e-10)


def test_StringAveragedKaczmarz_proximity_flag_false(get_full_variables):
    A, b = get_full_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0, 1, 2]], proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_StringAveragedKaczmarz_single_string_matches_sequential(get_full_variables):
    # one string covering all constraints is exactly the sequential sweep
    from suppy.feasibility import KaczmarzMethod

    A, b = get_full_variables
    alg_sa = StringAveragedKaczmarz(A, b, strings=[[0, 1, 2]])
    alg_s = KaczmarzMethod(A, b)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_sa.project(x.copy()), alg_s.project(x.copy()))


def test_StringAveragedKaczmarz_fejer_monotone(get_full_variables):
    A, b = get_full_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0, 1], [2]])
    z = np.array([1.0, 1.0])  # the solution point
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_StringAveragedKaczmarz_project_idempotent_on_solution(get_full_variables):
    A, b = get_full_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0, 1], [2]])
    x = np.array([1.0, 1.0])
    assert np.allclose(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_StringAveragedKaczmarz_solve_int_input(get_full_variables):
    A, b = get_full_variables
    alg = StringAveragedKaczmarz(A, b, strings=[[0, 1], [2]])
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_StringAveragedKaczmarz_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    strings = [[0, 1], [2]]
    x_np = StringAveragedKaczmarz(A, b, strings=strings).step(np.array([2.0, 2.0]))

    alg_cp = StringAveragedKaczmarz(cp.asarray(A), cp.asarray(b), strings=strings)
    x_cp = alg_cp.step(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
