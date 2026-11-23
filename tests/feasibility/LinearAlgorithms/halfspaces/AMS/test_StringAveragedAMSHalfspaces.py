import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import StringAveragedAMSHalfspace
from suppy.utils import LinearMapping


@pytest.fixture
def get_full_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    ub = -1 * np.array([-2, -2, -3 / 2, -3 / 2])
    return np.concatenate((A, -A)), np.concatenate((ub, ub))


@pytest.fixture
def get_sparse_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    A = sparse.csr_matrix(np.concatenate((A, -A)))
    ub = -1 * np.array([-2, -2, -3 / 2, -3 / 2])
    return A, np.concatenate((ub, ub))


def test_StringAveragedAMSHalfspace_constructor_full(get_full_variables):
    """Test the StringAveragedAMSHalfspace constructor."""
    A, b = get_full_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])  # sequential like

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert alg.strings == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0], [1], [2], [3], [4], [5], [6], [7]]
    )  # simultaneous like
    assert alg.strings == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert np.array_equal(alg.weights, np.ones(8) / 8)


def test_StringAveragedAMSHalfspace_constructor_sparse(get_sparse_variables):
    """Test the StringAveragedAMSHalfspace constructor."""
    A, b = get_sparse_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])  # sequential like

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert alg.strings == [[0, 1, 2, 3, 4, 5, 6, 7]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0], [1], [2], [3], [4], [5], [6], [7]]
    )  # simultaneous like
    assert alg.strings == [[0], [1], [2], [3], [4], [5], [6], [7]]
    assert np.array_equal(alg.weights, np.ones(8) / 8)


def test_StringAveragedAMSHalfspace_step_full_sequential_like(get_full_variables):
    """
    Test the step function of the StringAveragedAMSHalfspace class for
    sequential
    like strings.
    """

    A, b = get_full_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])  # sequential like

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([0, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([0, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_StringAveragedAMSHalfspace_step_sparse_sequential_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHalfspace class for
    sequential
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])  # sequential like

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([0, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([0, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_StringAveragedAMSHalfspace_step_full_simultaneous_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHalfspace class for
    simultaneous
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0], [1], [2], [3], [4], [5], [6], [7]]
    )  # simultaneous like

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([29 / 16, 29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.175, -1.175])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_StringAveragedAMSHalfspace_step_sparse_simultaneous_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHalfspace class for
    simultaneous
    like strings.
    """
    A, b = get_sparse_variables
    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0], [1], [2], [3], [4], [5], [6], [7]]
    )  # simultaneous like

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.175, 1.175])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([29 / 16, 29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.175, -1.175])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([-29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([29 / 16, -29 / 16])) < 1e-10)
    assert np.array_equal(x_n, x_5)


# ---------------------------------------------------------------------------
# StringAveragedAMSHalfspace: options and branches
# ---------------------------------------------------------------------------


def test_StringAveragedAMSHalfspace_weights_length_mismatch(get_full_variables):
    A, b = get_full_variables
    with pytest.raises(ValueError):
        StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3], [4, 5, 6, 7]], weights=[1.0])


def test_StringAveragedAMSHalfspace_custom_string_weights(get_full_variables):
    A, b = get_full_variables
    # strings [0] and [1] with weights 0.7/0.3 on x = [2, 2]:
    # string [0]: row 0 (x+y <= 2) violated by 2, ||a||^2 = 2 -> [1, 1];
    # string [1]: row 1 (-x+y <= 2) satisfied -> [2, 2];
    # average: 0.7*[1, 1] + 0.3*[2, 2] = [1.3, 1.3]
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0], [1]], weights=np.array([0.7, 0.3]))
    x_n = alg.step(np.array([2.0, 2.0]))
    assert np.all(abs(x_n - np.array([1.3, 1.3])) < 1e-10)


def test_StringAveragedAMSHalfspace_proximity_flag_false(get_full_variables):
    A, b = get_full_variables
    alg = StringAveragedAMSHalfspace(
        A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]], proximity_flag=False
    )
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_StringAveragedAMSHalfspace_single_string_matches_sequential(get_full_variables):
    # one string covering all constraints is exactly the sequential sweep
    from suppy.feasibility import SequentialAMSHalfspace

    A, b = get_full_variables
    alg_sa = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3, 4, 5, 6, 7]])
    alg_s = SequentialAMSHalfspace(A, b)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_sa.project(x.copy()), alg_s.project(x.copy()))


def test_StringAveragedAMSHalfspace_fejer_monotone(get_full_variables):
    A, b = get_full_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3], [4, 5, 6, 7]])
    z = np.zeros(2)  # interior point of the feasible set
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_StringAveragedAMSHalfspace_project_idempotent_on_feasible(get_full_variables):
    A, b = get_full_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3], [4, 5, 6, 7]])
    x = np.array([0.5, 0.5])
    assert np.array_equal(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_StringAveragedAMSHalfspace_solve_int_input(get_full_variables):
    A, b = get_full_variables
    alg = StringAveragedAMSHalfspace(A, b, strings=[[0, 1, 2, 3], [4, 5, 6, 7]])
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_StringAveragedAMSHalfspace_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    strings = [[0, 1, 2, 3], [4, 5, 6, 7]]
    x_np = StringAveragedAMSHalfspace(A, b, strings=strings).step(np.array([2.0, 2.0]))

    alg_cp = StringAveragedAMSHalfspace(cp.asarray(A), cp.asarray(b), strings=strings)
    x_cp = alg_cp.step(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
