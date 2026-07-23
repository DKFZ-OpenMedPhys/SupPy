import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialAMSHalfspace
from suppy.utils import LinearMapping


@pytest.fixture
def get_full_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    n, m = A.shape
    A_n = np.empty((2 * n, m), dtype=A.dtype)
    A_n[0::2] = A
    A_n[1::2] = -A

    ub = -1 * np.array([-2, -2, -3 / 2, -3 / 2])
    ub_n = np.empty(2 * n, dtype=ub.dtype)
    ub_n[0::2] = ub
    ub_n[1::2] = ub
    return A_n, ub_n


@pytest.fixture
def get_sparse_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    n, m = A.shape
    A_n = np.empty((2 * n, m), dtype=A.dtype)
    A_n[0::2] = A
    A_n[1::2] = -A
    A_n = sparse.csr_matrix(A_n)

    ub = -1 * np.array([-2, -2, -3 / 2, -3 / 2])
    ub_n = np.empty(2 * n, dtype=ub.dtype)
    ub_n[0::2] = ub
    ub_n[1::2] = ub
    return A_n, ub_n


@pytest.fixture
def get_SequentialAMSHalfspace_input_full(get_full_variables):
    A, b = get_full_variables
    return SequentialAMSHalfspace(A, b), A, b


@pytest.fixture
def get_SequentialAMSHalfspace_input_sparse(get_sparse_variables):
    A, b = get_sparse_variables
    return SequentialAMSHalfspace(A, b), A, b


def test_SequentialAMSHalfspace_no_relaxation_constructor_full(
    get_SequentialAMSHalfspace_input_full,
):
    """
    Test the SequentialAMSHalfspace constructor with no relaxation and a
    full
    matrix.
    """
    alg, A, b = get_SequentialAMSHalfspace_input_full

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_SequentialAMSHalfspace_no_relaxation_constructor_sparse(
    get_SequentialAMSHalfspace_input_sparse,
):
    """
    Test the SequentialAMSHalfspace constructor with no relaxation and a
    sparse
    matrix.
    """
    alg, A, b = get_SequentialAMSHalfspace_input_sparse

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.b, b)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_SequentialAMSHalfspace_constructor_wrong_bounds_shape(get_full_variables):
    """Test the SequentialAMSHalfspace constructor with wrong bounds shape."""
    A, _ = get_full_variables
    ub = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        SequentialAMSHalfspace(A, ub)


def test_SequentialAMSHalfspace_constructor_scalar_bounds(get_full_variables):
    """Test the SequentialAMSHyperslab constructor with scalar bounds."""
    A, _ = get_full_variables
    b = 1
    alg = SequentialAMSHalfspace(A, b)
    assert (alg.b == np.ones(8)).all()


def test_SequentialAMSHalfspace_map_full(get_SequentialAMSHalfspace_input_full):
    """Test the map function of the SequentialAMSHalfspace class with full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHalfspace_input_full

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, -2, 0, 0, 1, -1, 1, -1]))


def test_SequentialAMSHalfspace_map_sparse(get_SequentialAMSHalfspace_input_sparse):
    """Test the map function of the SequentialAMSHalfspace class with sparse
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHalfspace_input_sparse

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, -2, 0, 0, 1, -1, 1, -1]))


def test_SequentialAMSHalfspace_indexed_map_full(get_SequentialAMSHalfspace_input_full):
    """
    Test the indexed_map function of the SequentialAMSHalfspace class with
    full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHalfspace_input_full
    idx = [0, 2]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_SequentialAMSHalfspace_indexed_map_sparse(get_SequentialAMSHalfspace_input_sparse):
    """
    Test the indexed_map function of the SequentialAMSHalfspace class with
    sparse
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHalfspace_input_sparse
    idx = [0, 2]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_SequentialAMSHalfspace_step_full(get_SequentialAMSHalfspace_input_full):
    """Test the step function of the SequentialAMSHalfspace class with full
    matrix.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHalfspace_input_full

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


def test_SequentialAMSHalfspace_step_sparse(get_SequentialAMSHalfspace_input_sparse):
    """Test the step function of the SequentialAMSHalfspace class with
    sparse.
    """
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHalfspace_input_sparse

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


def test_SequentialAMSHalfspace_step_full_algoritimic_relaxation(get_full_variables):
    """Test the step function of the SequentialAMSHalfspace class with
    relaxation.
    """
    A, b = get_full_variables
    # test with relaxation
    alg = SequentialAMSHalfspace(A, b, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.5, 0.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_SequentialAMSHalfspace_step_sparse_algoritimic_relaxation(get_sparse_variables):
    """Test the step function of the SequentialAMSHalfspace class with
    relaxation.
    """
    A, b = get_sparse_variables
    # test with relaxation
    alg = SequentialAMSHalfspace(A, b, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.5, 0.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_SequentialAMSHalfspace_step_full_relaxation(get_full_variables):
    """Test the step function of the SequentialAMSHalfspace class with
    relaxation.
    """
    A, b = get_full_variables
    # test with relaxation
    alg = SequentialAMSHalfspace(A, b, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.5, 0.5])) < 1e-10)


def test_SequentialAMSHalfspace_step_sparse_relaxation(get_sparse_variables):
    """Test the step function of the SequentialAMSHalfspace class with
    relaxation.
    """
    A, b = get_sparse_variables
    # test with relaxation
    alg = SequentialAMSHalfspace(A, b, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.5, 0.5])) < 1e-10)


def test_SequentialAMSHalfspace_custom_cs(get_SequentialAMSHalfspace_input_full):
    """Test the step function of the SequentialAMSHalfspace class with
    custom.
    """
    _, A, b = get_SequentialAMSHalfspace_input_full

    # test with custom cs
    alg3 = SequentialAMSHalfspace(A, b, cs=[6, 4, 2, 0])

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg3.step(x_1)
    assert np.all(abs(x_n - np.array([1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg3.step(x_2)
    assert np.all(abs(x_n - np.array([3 / 4, 5 / 4])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg3.step(x_3)
    assert np.all(abs(x_n - np.array([0, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    # cs = [6, 4, 2, 0] only visits the *upper* halfspace of each pair, so the
    # lower bounds (rows 5 and 7) are never enforced:
    # x_4 = [-2, 2]: row 6 (y <= 1.5) -> [-2, 1.5]; row 4 ok (x >= -1.5 is row
    # 5, not visited); row 2 (-x + y <= 2): viol 1.5, ||a||^2 = 2 ->
    # -(1.5/2)*[-1, 1] -> [-1.25, 0.75]; row 0 ok.
    x_n = alg3.step(x_4)
    assert np.all(abs(x_n - np.array([-1.25, 0.75])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    # x_5 = [-3, 0]: rows 6 and 4 ok; row 2: viol 1 -> -(1/2)*[-1, 1] ->
    # [-2.5, -0.5]; row 0 ok.
    x_n = alg3.step(x_5)
    assert np.all(abs(x_n - np.array([-2.5, -0.5])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SequentialAMSHalfspace_infinity_bounds(get_SequentialAMSHalfspace_input_full):
    """
    Test the step function of the SequentialAMSHalfspace class with infinity
    in
    bounds.
    """
    _, A, _ = get_SequentialAMSHalfspace_input_full

    # test with infinity in bounds
    b2 = np.array([np.inf, 2, 2, 2, 3 / 2, 3 / 2, 3 / 2, 3 / 2])
    alg4 = SequentialAMSHalfspace(A, b2)

    x_1 = np.array([2.0, 2.0])
    x_2 = np.array([1.0, 3.0])
    x_3 = np.array([0.0, 3.0])
    x_4 = np.array([-2.0, 2.0])
    x_5 = np.array([-3.0, 0.0])

    x_n = alg4.step(x_1)
    assert np.all(abs(x_n - np.array([1.5, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)

    x_n = alg4.step(x_2)
    assert np.all(abs(x_n - np.array([1, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_2)

    x_n = alg4.step(x_3)
    assert np.all(abs(x_n - np.array([0.5, 1.5])) < 1e-10)
    assert np.array_equal(x_n, x_3)

    x_n = alg4.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg4.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SequentialAMSHalfspace_proximity(get_SequentialAMSHalfspace_input_full):
    """Test the proximity function of the SequentialAMSHalfspace class."""
    (
        alg,
        _,
        _,
    ) = get_SequentialAMSHalfspace_input_full

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

    assert np.abs(prox_1[0] - 0.02) < 1e-10
    assert np.abs(prox_1[1] - 0.4) < 1e-10

    assert np.abs(prox_2[0] - 9 / 16) < 1e-10
    assert np.abs(prox_2[1] - 2) < 1e-10

    assert np.abs(prox_3[0] - 0.02) < 1e-10
    assert np.abs(prox_3[1] - 0.4) < 1e-10

    assert np.abs(prox_4[0] - 9 / 16) < 1e-10
    assert np.abs(prox_4[1] - 2) < 1e-10

    assert np.abs(prox_5[0] - 9 / 16) < 1e-10
    assert np.abs(prox_5[1] - 2) < 1e-10


# ---------------------------------------------------------------------------
# SequentialAMSHalfspace: options and branches
# ---------------------------------------------------------------------------


def test_SequentialAMSHalfspace_proximity_flag_false(get_full_variables):
    A, b = get_full_variables
    alg = SequentialAMSHalfspace(A, b, proximity_flag=False)
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


def test_SequentialAMSHalfspace_invalid_proximity_measure(
    get_SequentialAMSHalfspace_input_full,
):
    alg, _, _ = get_SequentialAMSHalfspace_input_full
    x = np.array([2.0, 2.0])
    with pytest.raises(ValueError):
        alg.proximity(x, [("q_norm", 2)])
    with pytest.raises(ValueError):
        alg.proximity(x, ["sup_norm"])


# ---------------------------------------------------------------------------
# solve() machinery (subset; the full battery lives in the hyperslab tests)
# ---------------------------------------------------------------------------


def test_SequentialAMSHalfspace_solve_storage_iters_list(
    get_SequentialAMSHalfspace_input_full,
):
    alg, _, _ = get_SequentialAMSHalfspace_input_full
    alg.solve(
        np.array([2.0, 2.0]),
        max_iter=100,
        prox_tol=0.0,
        del_prox_tol=1e-8,
        del_prox_n=3,
        storage=True,
        storage_iters=[0, 2],
    )
    # only the requested iteration indices are kept (0 = initial point)
    assert alg.all_x.shape == (2, 2)
    assert np.array_equal(alg.all_x[0], np.array([2.0, 2.0]))
    assert np.allclose(alg.all_x[1], np.array([1.0, 1.0]))


def test_SequentialAMSHalfspace_solve_alternative_initial_call(
    get_SequentialAMSHalfspace_input_full,
):
    alg, _, _ = get_SequentialAMSHalfspace_input_full
    x_0 = np.array([2.0, 2.0])
    x_n = alg.solve(
        x_0.copy(),
        max_iter=100,
        alternative_stopping_criterion_initial_call=lambda x, alg_: True,
    )
    # an initial call returning True stops the run before any projection
    assert np.array_equal(x_n, x_0)
    assert len(alg.proximities) == 1


def test_SequentialAMSHalfspace_solve_max_iter_zero(get_SequentialAMSHalfspace_input_full):
    alg, _, _ = get_SequentialAMSHalfspace_input_full
    x_0 = np.array([2.0, 2.0])
    x_n = alg.solve(x_0.copy(), max_iter=0, storage=True)
    assert np.array_equal(x_n, x_0)
    assert alg.proximities.shape == (1, 1)
    assert alg.all_x.shape == (1, 2)


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_SequentialAMSHalfspace_matches_hyperslab_formulation(get_full_variables):
    # the interleaved halfspace pairs (upper, lower) encode the same problem
    # as the hyperslab formulation and are visited in the same order, so the
    # exact projections agree
    from suppy.feasibility import SequentialAMSHyperslab

    A_hs, b = get_full_variables
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    ub = np.array([2.0, 2.0, 1.5, 1.5])
    alg_hs = SequentialAMSHalfspace(A_hs, b)
    alg_slab = SequentialAMSHyperslab(A, -ub, ub)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(alg_hs.project(x.copy()), alg_slab.project(x.copy()))


def test_SequentialAMSHalfspace_fejer_monotone(get_full_variables):
    A, b = get_full_variables
    alg = SequentialAMSHalfspace(A, b)
    z = np.zeros(2)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_SequentialAMSHalfspace_project_idempotent_on_feasible(get_full_variables):
    A, b = get_full_variables
    alg = SequentialAMSHalfspace(A, b)
    x = np.array([0.5, 0.5])
    assert np.array_equal(alg.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SequentialAMSHalfspace_solve_int_input(get_SequentialAMSHalfspace_input_full):
    alg, _, _ = get_SequentialAMSHalfspace_input_full
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_SequentialAMSHalfspace_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, b = get_full_variables
    x_np = SequentialAMSHalfspace(A, b).step(np.array([2.0, 2.0]))

    alg_cp = SequentialAMSHalfspace(cp.asarray(A), cp.asarray(b))
    x_cp = alg_cp.step(cp.asarray([2.0, 2.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------


def test_SequentialAMSHalfspace_solve_to_feasibility(get_full_variables):
    A, b = get_full_variables
    alg = SequentialAMSHalfspace(A, b)
    x_n = alg.solve(np.array([3.0, -2.5]), max_iter=500)
    assert np.all(A @ x_n <= b + 1e-3)
