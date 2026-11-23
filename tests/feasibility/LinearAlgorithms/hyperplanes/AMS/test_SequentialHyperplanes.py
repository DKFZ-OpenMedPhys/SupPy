import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import KaczmarzMethod
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
def get_KaczmarzMethod_input_full(get_full_variables):
    A, b = get_full_variables
    return KaczmarzMethod(A, b), A, b


@pytest.fixture
def get_KaczmarzMethod_input_sparse(get_sparse_variables):
    A, b = get_sparse_variables
    return KaczmarzMethod(A, b), A, b


def test_KaczmarzMethod_no_relaxation_constructor_full(
    get_KaczmarzMethod_input_full,
):
    """
    Test the KaczmarzMethod constructor with no relaxation and a
    full
    matrix.
    """
    alg, A, b = get_KaczmarzMethod_input_full

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_KaczmarzMethod_no_relaxation_constructor_sparse(
    get_KaczmarzMethod_input_sparse,
):
    """
    Test the KaczmarzMethod constructor with no relaxation and a
    sparse
    matrix.
    """
    alg, A, b = get_KaczmarzMethod_input_sparse

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_KaczmarzMethod_constructor_wrong_bounds_shape(get_full_variables):
    """Test the SequentialAMSHyperslab constructor with wrong bounds shape."""
    A, _ = get_full_variables
    ub = np.array([1, 2])
    with pytest.raises(ValueError):
        KaczmarzMethod(A, ub)


def test_KaczmarzMethod_constructor_scalar_bounds(get_full_variables):
    """Test the SequentialAMSHyperslab constructor with scalar bounds."""
    A, _ = get_full_variables
    b = 1
    alg = KaczmarzMethod(A, b)
    assert (alg.b == np.array([1, 1, 1])).all()


def test_KaczmarzMethod_map_full(get_KaczmarzMethod_input_full):
    """Test the map function of the KaczmarzMethod class with full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_KaczmarzMethod_input_full

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([1, 0, 1]))


def test_KaczmarzMethod_map_sparse(get_KaczmarzMethod_input_sparse):
    """Test the map function of the KaczmarzMethod class with sparse
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_KaczmarzMethod_input_sparse

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([1, 0, 1]))


def test_KaczmarzMethod_indexed_map_full(get_KaczmarzMethod_input_full):
    """
    Test the indexed_map function of the KaczmarzMethod class with
    full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_KaczmarzMethod_input_full
    idx = [0, 2]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([1, 1]))


def test_KaczmarzMethod_indexed_map_sparse(get_KaczmarzMethod_input_sparse):
    """
    Test the indexed_map function of the KaczmarzMethod class with
    sparse
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_KaczmarzMethod_input_sparse
    idx = [0, 2]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([1, 1]))


def test_KaczmarzMethod_step_full(get_KaczmarzMethod_input_full):
    """Test the step function of the KaczmarzMethod class with full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_KaczmarzMethod_input_full

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


def test_KaczmarzMethod_step_sparse(get_KaczmarzMethod_input_sparse):
    """Test the step function of the KaczmarzMethod class with
    sparse.
    """
    (
        alg,
        _,
        _,
    ) = get_KaczmarzMethod_input_sparse

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


def test_KaczmarzMethod_step_full_algoritimic_relaxation(get_full_variables):
    """Test the step function of the KaczmarzMethod class with
    relaxation.
    """
    A, b = get_full_variables
    # test with relaxation
    alg = KaczmarzMethod(A, b, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.625, 1.0625])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_KaczmarzMethod_step_sparse_algoritimic_relaxation(get_sparse_variables):
    """Test the step function of the KaczmarzMethod class with
    relaxation.
    """
    A, b = get_sparse_variables
    # test with relaxation
    alg = KaczmarzMethod(A, b, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.625, 1.0625])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_KaczmarzMethod_step_full_relaxation(get_full_variables):
    """Test the step function of the KaczmarzMethod class with
    relaxation.
    """
    A, b = get_full_variables
    # test with relaxation
    alg = KaczmarzMethod(A, b, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.25, 0.5])) < 1e-10)


def test_KaczmarzMethod_step_sparse_relaxation(get_sparse_variables):
    """Test the step function of the KaczmarzMethod class with
    relaxation.
    """
    A, b = get_sparse_variables
    # test with relaxation
    alg = KaczmarzMethod(A, b, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1.25, 0.5])) < 1e-10)


def test_KaczmarzMethod_custom_cs(get_KaczmarzMethod_input_full):
    """Test the step function of the KaczmarzMethod class with
    custom.
    """
    _, A, b = get_KaczmarzMethod_input_full

    # test with custom cs
    alg = KaczmarzMethod(A, b, cs=[2, 1, 0])

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    # check that project gives the same result
    x_1 = np.array([2.0, 2.0])
    x_proj = alg.project(x_1)
    assert np.all(abs(x_proj - np.array([1, 1.5])) < 1e-10)
    assert np.array_equal(x_proj, x_1)
    assert np.array_equal(x_proj, x_n)

    # cs = [2, 1, 0] visits the hyperplanes y = 1, x = y, x = 1 in that order.
    # x_2 = [1, 3]: y=1 -> [1, 1]; x=y already satisfied; x=1 already satisfied.
    x_n = alg.step(x_2)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    # x_3 = [0, 3]: y=1 -> [0, 1]; x=y -> [0.5, 0.5]; x=1 -> [1, 0.5].
    x_n = alg.step(x_3)
    assert np.all(abs(x_n - np.array([1, 0.5])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    # x_4 = [-2, 2]: y=1 -> [-2, 1]; x=y -> [-0.5, -0.5]; x=1 -> [1, -0.5].
    x_n = alg.step(x_4)
    assert np.all(abs(x_n - np.array([1, -0.5])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    # x_5 = [-3, 0]: y=1 -> [-3, 1]; x=y -> [-1, -1]; x=1 -> [1, -1].
    x_n = alg.step(x_5)
    assert np.all(abs(x_n - np.array([1, -1])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_KaczmarzMethod_proximity(get_KaczmarzMethod_input_full):
    """Test the proximity function of the KaczmarzMethod class."""
    (
        alg,
        _,
        _,
    ) = get_KaczmarzMethod_input_full

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
# KaczmarzMethod: options and branches
# ---------------------------------------------------------------------------


def test_KaczmarzMethod_proximity_flag_false(get_full_variables):
    A, b = get_full_variables
    alg = KaczmarzMethod(A, b, proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


def test_KaczmarzMethod_invalid_proximity_measure(get_KaczmarzMethod_input_full):
    alg, _, _ = get_KaczmarzMethod_input_full
    with pytest.raises(ValueError):
        alg.proximity(np.array([2.0, 2.0]), ["not_a_measure"])


def test_KaczmarzMethod_solve_max_iter_zero(get_KaczmarzMethod_input_full):
    alg, _, _ = get_KaczmarzMethod_input_full
    x_n = alg.solve(np.array([2.0, 2.0]), max_iter=0)
    # no projection happened; only the initial proximity was recorded
    assert np.array_equal(x_n, np.array([2.0, 2.0]))
    assert len(alg.proximities) == 1


def test_KaczmarzMethod_solve_storage_iters_int(get_KaczmarzMethod_input_full):
    alg, _, _ = get_KaczmarzMethod_input_full
    # sweep 1: [2, 2] -> [1.5, 1]; sweep 2: [1.5, 1] -> [1, 1] (exact
    # solution, proximity 0 -> stop). storage_iters=2 stores indices 0 and 2.
    x_n = alg.solve(np.array([2.0, 2.0]), max_iter=10, storage=True, storage_iters=2)
    assert np.allclose(x_n, np.array([1.0, 1.0]))
    assert alg.all_x.shape == (2, 2)
    assert np.allclose(alg.all_x[0], np.array([2.0, 2.0]))
    assert np.allclose(alg.all_x[1], np.array([1.0, 1.0]))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_KaczmarzMethod_solve_converges_to_solution(get_full_variables):
    # the system is consistent with unique solution [1, 1]
    A, b = get_full_variables
    rng = np.random.default_rng(42)
    for _ in range(5):
        alg = KaczmarzMethod(A, b)
        x = 4 * rng.standard_normal(2)
        x_n = alg.solve(x, max_iter=500)
        assert np.allclose(x_n, np.array([1.0, 1.0]), atol=1e-4)


def test_KaczmarzMethod_fejer_monotone(get_full_variables):
    A, b = get_full_variables
    alg = KaczmarzMethod(A, b)
    z = np.array([1.0, 1.0])  # the solution point
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_KaczmarzMethod_project_idempotent_on_solution(get_full_variables):
    A, b = get_full_variables
    alg = KaczmarzMethod(A, b)
    x = np.array([1.0, 1.0])
    assert np.allclose(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_KaczmarzMethod_solve_int_input(get_full_variables):
    A, b = get_full_variables
    alg = KaczmarzMethod(A, b)
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_KaczmarzMethod_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    x_np = KaczmarzMethod(A, b).step(np.array([2.0, 2.0]))

    alg_cp = KaczmarzMethod(cp.asarray(A), cp.asarray(b))
    x_cp = alg_cp.step(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
