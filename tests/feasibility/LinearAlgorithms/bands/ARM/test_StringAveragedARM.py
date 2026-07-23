import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import StringAveragedARM
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


def test_StringAveragedARM_constructor_full(get_ARM_variables_full):
    """
    Test the constructor of the StringAveragedARM class with full
    matrices.
    """
    A, lb, ub = get_ARM_variables_full
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]])  # sequential like
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert alg.strings == [[0, 1]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])  # simultaneous like
    assert alg.strings == [[0], [1]]
    assert np.array_equal(alg.weights, np.ones(2) / 2)


def test_StringAveragedARM_constructor_sparse(get_ARM_variables_sparse):
    """
    Test the constructor of the StringAveragedARM class with sparse
    matrices.
    """
    A, lb, ub = get_ARM_variables_sparse
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]])  # sequential like
    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert alg.strings == [[0, 1]]
    assert np.array_equal(alg.weights, np.ones(1))
    assert alg.relaxation == 1.0
    assert alg.algorithmic_relaxation == 1.0

    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])  # simultaneous like
    assert alg.strings == [[0], [1]]
    assert np.array_equal(alg.weights, np.ones(2) / 2)


def test_StringAveragedARM_step_full_sequential_like(get_ARM_variables_full):
    """
    Test the step method of the StringAveragedARM class with full matrices
    for sequential like strings.
    """
    A, lb, ub = get_ARM_variables_full
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]])  # sequential like

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


def test_StringAveragedARM_step_sparse_sequential_like(get_ARM_variables_sparse):
    """
    Test the step method of the StringAveragedARM class with sparse matrices
    for sequential like strings.
    """
    A, lb, ub = get_ARM_variables_sparse
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]])  # sequential like

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


def test_StringAveragedARM_step_full_simultaneous_like(get_ARM_variables_sparse):
    """
    Test the step method of the StringAveragedARM class with full matrices
    for simultaneous like strings.
    """
    A, lb, ub = get_ARM_variables_sparse

    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])  # simultaneous like

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


def test_StringAveragedARM_step_sparse_simultaneous_like(get_ARM_variables_sparse):
    """
    Test the step method of the StringAveragedARM class with sparse matrices
    for simultaneous like strings.
    """
    A, lb, ub = get_ARM_variables_sparse

    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])  # simultaneous like

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


# ---------------------------------------------------------------------------
# StringAveragedARM: options and branches
# ---------------------------------------------------------------------------


def test_StringAveragedARM_weights_length_mismatch(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    with pytest.raises(ValueError):
        StringAveragedARM(A, lb, ub, strings=[[0], [1]], weights=[1.0])


def test_StringAveragedARM_custom_string_weights(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    # strings [0] and [1] with weights 0.7/0.3 on x = [1, 1]:
    # string [0]: band 0 violated (d = 2) -> [1, 1] - 0.375*[1, 1] = [0.625, 0.625];
    # string [1]: band 1 satisfied -> [1, 1];
    # average: 0.7*[0.625, 0.625] + 0.3*[1, 1] = [0.7375, 0.7375]
    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]], weights=np.array([0.7, 0.3]))
    x_n = alg.step(np.array([1.0, 1.0]))
    assert np.all(abs(x_n - np.array([0.7375, 0.7375])) < 1e-10)


def test_StringAveragedARM_proximity_flag_false(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = StringAveragedARM(A, lb, ub, strings=[[0, 1]], proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_StringAveragedARM_single_string_matches_sequential(get_ARM_variables_full):
    # one string covering all constraints is exactly the sequential sweep
    from suppy.feasibility import SequentialARM

    A, lb, ub = get_ARM_variables_full
    alg_sa = StringAveragedARM(A, lb, ub, strings=[[0, 1]])
    alg_s = SequentialARM(A, lb, ub)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_sa.project(x.copy()), alg_s.project(x.copy()))


def test_StringAveragedARM_fejer_monotone(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])
    z = np.zeros(2)  # interior point of the feasible set
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_StringAveragedARM_project_idempotent_on_feasible(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])
    x = np.array([0.0, 0.0])
    assert np.array_equal(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_StringAveragedARM_solve_int_input(get_ARM_variables_full):
    A, lb, ub = get_ARM_variables_full
    alg = StringAveragedARM(A, lb, ub, strings=[[0], [1]])
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([1, 1]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_StringAveragedARM_cupy_backend(get_ARM_variables_full):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_ARM_variables_full
    strings = [[0], [1]]
    x_np = StringAveragedARM(A, lb, ub, strings=strings).step(np.array([1.0, 1.0]))

    alg_cp = StringAveragedARM(cp.asarray(A), cp.asarray(lb), cp.asarray(ub), strings=strings)
    x_cp = alg_cp.step(cp.asarray([1.0, 1.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
