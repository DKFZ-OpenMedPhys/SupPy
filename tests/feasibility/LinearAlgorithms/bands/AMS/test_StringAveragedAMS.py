import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import StringAveragedAMSHyperslab
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


def test_StringAveragedAMSHyperslab_constructor_full(get_full_variables):
    """Test the StringAveragedAMSHyperslab constructor."""
    A, lb, ub = get_full_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1, 2, 3]])  # sequential like

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert alg.strings == [[0, 1, 2, 3]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0], [1], [2], [3]])  # simultaneous like
    assert alg.strings == [[0], [1], [2], [3]]
    assert np.array_equal(alg.weights, np.ones(4) / 4)


def test_StringAveragedAMSHyperslab_constructor_sparse(get_sparse_variables):
    """Test the StringAveragedAMSHyperslab constructor."""
    A, lb, ub = get_sparse_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1, 2, 3]])  # sequential like

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert alg.strings == [[0, 1, 2, 3]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0], [1], [2], [3]])  # simultaneous like
    assert alg.strings == [[0], [1], [2], [3]]
    assert np.array_equal(alg.weights, np.ones(4) / 4)


def test_StringAveragedAMSHyperslab_step_full_sequential_like(get_full_variables):
    """
    Test the step function of the StringAveragedAMSHyperslab class for
    sequential
    like strings.
    """

    A, lb, ub = get_full_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1, 2, 3]])  # sequential like

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


def test_StringAveragedAMSHyperslab_step_sparse_sequential_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHyperslab class for
    sequential
    like strings.
    """
    A, lb, ub = get_sparse_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1, 2, 3]])  # sequential like

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


def test_StringAveragedAMSHyperslab_step_full_simultaneous_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHyperslab class for
    simultaneous
    like strings.
    """
    A, lb, ub = get_sparse_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0], [1], [2], [3]])  # simultaneous like

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.15, 1.15])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.15, 1.15])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - 1 / 4 * np.array([6.5, 6.5])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.15, -1.15])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - 1 / 4 * np.array([-6.5, -6.5])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - 1 / 4 * np.array([6.5, -6.5])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_StringAveragedAMSHyperslab_step_sparse_simultaneous_like(get_sparse_variables):
    """
    Test the step function of the StringAveragedAMSHyperslab class for
    simultaneous
    like strings.
    """
    A, lb, ub = get_sparse_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0], [1], [2], [3]])  # simultaneous like

    x_1 = np.array([1.2, 1.2])
    x_2 = np.array([2.0, 2.0])
    x_3 = np.array([-1.2, -1.2])
    x_4 = np.array([-2.0, -2.0])
    x_5 = np.array([2.0, -2.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.15, 1.15])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result as step
    x_1 = np.array([1.2, 1.2])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1.15, 1.15])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    x_n = alg.step(x_2)
    assert np.all(abs(x_n - 1 / 4 * np.array([6.5, 6.5])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([-1.15, -1.15])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg.step(x_4)
    assert np.all(abs(x_n - 1 / 4 * np.array([-6.5, -6.5])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg.step(x_5)
    assert np.all(abs(x_n - 1 / 4 * np.array([6.5, -6.5])) < 1e-10)
    assert np.array_equal(x_n, x_5)


# ---------------------------------------------------------------------------
# StringAveragedAMSHyperslab: options and branches
# ---------------------------------------------------------------------------


def test_StringAveragedAMSHyperslab_weights_length_mismatch(get_full_variables):
    A, lb, ub = get_full_variables
    with pytest.raises(ValueError):
        StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1], [2, 3]], weights=[1.0])


def test_StringAveragedAMSHyperslab_custom_string_weights(get_full_variables):
    A, lb, ub = get_full_variables
    # strings [0] and [1] with weights 0.7/0.3 on x = [2, 2]:
    # string [0]: row 0 (x+y <= 2) violated by 2, ||a||^2 = 2 -> [1, 1];
    # string [1]: row 1 (-x+y in [-2, 2]) satisfied  -> [2, 2];
    # average: 0.7*[1, 1] + 0.3*[2, 2] = [1.3, 1.3]
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0], [1]], weights=np.array([0.7, 0.3]))
    x_n = alg.step(np.array([2.0, 2.0]))
    assert np.all(abs(x_n - np.array([1.3, 1.3])) < 1e-10)


def test_StringAveragedAMSHyperslab_proximity_flag_false(get_full_variables):
    A, lb, ub = get_full_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1, 2, 3]], proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_StringAveragedAMSHyperslab_single_string_matches_sequential(get_full_variables):
    # one string covering all constraints is exactly the sequential sweep
    from suppy.feasibility import SequentialAMSHyperslab

    A, lb, ub = get_full_variables
    alg_sa = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1, 2, 3]])
    alg_s = SequentialAMSHyperslab(A, lb, ub)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_sa.project(x.copy()), alg_s.project(x.copy()))


def test_StringAveragedAMSHyperslab_fejer_monotone(get_full_variables):
    A, lb, ub = get_full_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1], [2, 3]])
    z = np.zeros(2)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_StringAveragedAMSHyperslab_project_idempotent_on_feasible(get_full_variables):
    A, lb, ub = get_full_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1], [2, 3]])
    x = np.array([0.5, 0.5])
    assert np.array_equal(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_StringAveragedAMSHyperslab_solve_int_input(get_full_variables):
    A, lb, ub = get_full_variables
    alg = StringAveragedAMSHyperslab(A, lb, ub, strings=[[0, 1], [2, 3]])
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_StringAveragedAMSHyperslab_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_full_variables
    strings = [[0, 1], [2, 3]]
    x_np = StringAveragedAMSHyperslab(A, lb, ub, strings=strings).step(np.array([2.0, 2.0]))

    alg_cp = StringAveragedAMSHyperslab(
        cp.asarray(A), cp.asarray(lb), cp.asarray(ub), strings=strings
    )
    x_cp = alg_cp.step(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
