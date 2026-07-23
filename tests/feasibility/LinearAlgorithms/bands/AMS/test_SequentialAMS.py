import numpy as np
import pytest
import scipy.sparse as sparse
from suppy.feasibility import SequentialAMSHyperslab
from suppy.perturbations import PowerSeriesGradientPerturbation
from suppy.superiorization import Superiorization
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


@pytest.fixture
def get_SequentialAMSHyperslab_input_full(get_full_variables):
    A, lb, ub = get_full_variables
    return SequentialAMSHyperslab(A, lb, ub), A, lb, ub


@pytest.fixture
def get_SequentialAMSHyperslab_input_sparse(get_sparse_variables):
    A, lb, ub = get_sparse_variables
    return SequentialAMSHyperslab(A, lb, ub), A, lb, ub


def test_SequentialAMSHyperslab_no_relaxation_constructor_full(
    get_SequentialAMSHyperslab_input_full,
):
    """
    Test the SequentialAMSHyperslab constructor with no relaxation and a
    full
    matrix.
    """
    alg, A, lb, ub = get_SequentialAMSHyperslab_input_full

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A, A)
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(len(A)))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_SequentialAMSHyperslab_no_relaxation_constructor_sparse(
    get_SequentialAMSHyperslab_input_sparse,
):
    """
    Test the SequentialAMSHyperslab constructor with no relaxation and a
    sparse
    matrix.
    """
    alg, A, lb, ub = get_SequentialAMSHyperslab_input_sparse

    assert isinstance(alg.A, LinearMapping)
    assert np.array_equal(alg.A.todense(), A.todense())
    assert np.array_equal(alg.bounds.l, lb)
    assert np.array_equal(alg.bounds.u, ub)
    assert np.array_equal(alg.cs, np.arange(A.shape[0]))
    assert alg.algorithmic_relaxation == 1.0
    assert alg.relaxation == 1.0


def test_SequentialAMSHyperslab_constructor_wrong_bounds_shape(get_full_variables):
    """Test the SequentialAMSHyperslab constructor with wrong bounds shape."""
    A, lb, _ = get_full_variables
    ub = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        SequentialAMSHyperslab(A, lb, ub)

    lb = np.array([1, 2, 3])
    with pytest.raises(ValueError):
        SequentialAMSHyperslab(A, lb, ub)


def test_SequentialAMSHyperslab_map_full(get_SequentialAMSHyperslab_input_full):
    """Test the map function of the SequentialAMSHyperslab class with full
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, 0, 1, 1]))


def test_SequentialAMSHyperslab_map_sparse(get_SequentialAMSHyperslab_input_sparse):
    """Test the map function of the SequentialAMSHyperslab class with sparse
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_sparse

    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.map(x_map), np.array([2, 0, 1, 1]))


def test_SequentialAMSHyperslab_indexed_map_full(get_SequentialAMSHyperslab_input_full):
    """
    Test the indexed_map function of the SequentialAMSHyperslab class with
    full
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_SequentialAMSHyperslab_indexed_map_sparse(get_SequentialAMSHyperslab_input_sparse):
    """
    Test the indexed_map function of the SequentialAMSHyperslab class with
    sparse
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_sparse
    idx = [0, 1]
    # test map function(s)
    x_map = np.array([1, 1])
    assert np.array_equal(alg.indexed_map(x_map, idx), np.array([2, 0]))


def test_SequentialAMSHyperslab_step_full(get_SequentialAMSHyperslab_input_full):
    """Test the step function of the SequentialAMSHyperslab class with full
    matrix.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full

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


def test_SequentialAMSHyperslab_step_sparse(get_SequentialAMSHyperslab_input_sparse):
    """Test the step function of the SequentialAMSHyperslab class with
    sparse.
    """
    alg, _, _, _ = get_SequentialAMSHyperslab_input_sparse

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


def test_SequentialAMSHyperslab_step_full_algoritimic_relaxation(get_full_variables):
    """Test the step function of the SequentialAMSHyperslab class with
    relaxation.
    """
    A, lb, ub = get_full_variables
    # test with relaxation
    alg = SequentialAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.5, 0.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_SequentialAMSHyperslab_step_sparse_algoritimic_relaxation(get_sparse_variables):
    """Test the step function of the SequentialAMSHyperslab class with
    relaxation.
    """
    A, lb, ub = get_sparse_variables
    # test with relaxation
    alg = SequentialAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5)
    assert alg.algorithmic_relaxation == 1.5
    assert alg.relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.5, 0.5])) < 1e-10)
    assert np.array_equal(x_n, x_1)


def test_SequentialAMSHyperslab_step_full_relaxation(get_full_variables):
    """Test the step function of the SequentialAMSHyperslab class with
    relaxation.
    """
    A, lb, ub = get_full_variables
    # test with relaxation
    alg = SequentialAMSHyperslab(A, lb, ub, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.5, 0.5])) < 1e-10)


def test_SequentialAMSHyperslab_step_sparse_relaxation(get_sparse_variables):
    """Test the step function of the SequentialAMSHyperslab class with
    relaxation.
    """
    A, lb, ub = get_sparse_variables
    # test with relaxation
    alg = SequentialAMSHyperslab(A, lb, ub, relaxation=1.5)
    assert alg.relaxation == 1.5
    assert alg.algorithmic_relaxation == 1.0

    x_1 = np.array([2.0, 2.0])
    x_n = alg.step(x_1)
    assert np.all(abs(x_n - np.array([0.5, 0.5])) < 1e-10)


def test_SequentialAMSHyperslab_custom_cs(get_SequentialAMSHyperslab_input_full):
    """Test the step function of the SequentialAMSHyperslab class with
    custom.
    """
    _, A, lb, ub = get_SequentialAMSHyperslab_input_full

    # test with custom cs
    alg3 = SequentialAMSHyperslab(A, lb, ub, cs=[3, 2, 1, 0])

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

    x_n = alg3.step(x_4)
    assert np.all(abs(x_n - np.array([-1, 1])) < 1e-10)
    assert np.array_equal(x_n, x_4)

    x_n = alg3.step(x_5)
    assert np.all(abs(x_n - np.array([-1.5, 0])) < 1e-10)
    assert np.array_equal(x_n, x_5)


def test_SequentialAMSHyperslab_infinity_bounds(get_SequentialAMSHyperslab_input_full):
    """
    Test the step function of the SequentialAMSHyperslab class with infinity
    in
    bounds.
    """
    _, A, lb, _ = get_SequentialAMSHyperslab_input_full

    # test with infinity in bounds
    ub2 = np.array([np.inf, 2, 3 / 2, 3 / 2])
    alg4 = SequentialAMSHyperslab(A, lb, ub2)

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


def test_SequentialAMSHyperslab_proximity(get_SequentialAMSHyperslab_input_full):
    """Test the proximity function of the SequentialAMSHyperslab class."""
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full

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


def test_SequentialAMS_solve(get_SequentialAMSHyperslab_input_full):
    """Test the solve function of the SequentialAMSHyperslab class."""
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    x_1 = np.array([2.0, 2.0])
    x_n = alg.solve(x_1)
    assert np.all(np.abs(x_n - np.array([1.0, 1.0])) < 1e-10)


# ---------------------------------------------------------------------------
# SequentialAMSHyperslab: options and branches
# ---------------------------------------------------------------------------


def test_SequentialAMSHyperslab_proximity_flag_false(get_full_variables):
    A, lb, ub = get_full_variables
    alg = SequentialAMSHyperslab(A, lb, ub, proximity_flag=False)
    # the flag disables the object for proximity calculations: zeros are
    # returned regardless of how infeasible x is
    prox = alg.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


def test_SequentialAMSHyperslab_invalid_proximity_measure(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    x = np.array([2.0, 2.0])
    with pytest.raises(ValueError):
        alg.proximity(x, [("q_norm", 2)])
    with pytest.raises(ValueError):
        alg.proximity(x, ["sup_norm"])


# ---------------------------------------------------------------------------
# solve() machinery
# ---------------------------------------------------------------------------


def test_SequentialAMSHyperslab_solve_prox_tol_early_exit(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    # [2, 2] projects to the feasible point [1, 1] in one step, so the
    # proximity criterion fires immediately: one iteration, two proximity
    # entries (initial + after iteration 1)
    x_n = alg.solve(np.array([2.0, 2.0]), max_iter=100, prox_tol=1e-6)
    assert np.all(np.abs(x_n - np.array([1.0, 1.0])) < 1e-10)
    assert len(alg.proximities) == 2


def test_SequentialAMSHyperslab_solve_stall_exit(get_SequentialAMSHyperslab_input_full):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    # prox_tol=0 is unreachable (proximity >= 0 is never < 0), so the run can
    # only stop via the stall criterion: iteration 1 improves the proximity
    # (9/8 -> 0, resets the counter), iterations 2-4 leave it unchanged
    # (delta 0 < del_prox_tol) -> _n_tol reaches del_prox_n=3 -> 4 iterations
    x_n = alg.solve(
        np.array([2.0, 2.0]), max_iter=100, prox_tol=0.0, del_prox_tol=1e-8, del_prox_n=3
    )
    assert np.all(np.abs(x_n - np.array([1.0, 1.0])) < 1e-10)
    assert alg._n_tol == 3
    assert len(alg.proximities) == 5


def test_SequentialAMSHyperslab_solve_storage(get_SequentialAMSHyperslab_input_full):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    x_0 = np.array([2.0, 2.0])
    x_n = alg.solve(x_0.copy(), max_iter=100, storage=True)
    # every iterate is stored: row 0 is the start, the last row the result
    assert isinstance(alg.all_x, np.ndarray)
    assert alg.all_x.shape == (2, 2)
    assert np.array_equal(alg.all_x[0], x_0)
    assert np.allclose(alg.all_x[-1], x_n)
    # stored rows are copies, not aliases of the returned iterate
    x_n[0] = 100.0
    assert alg.all_x[-1][0] != 100.0


def test_SequentialAMSHyperslab_solve_storage_iters_int(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    # the stall run performs 4 iterations (see the stall test);
    # storage_iters=2 keeps indices 0, 2 and 4 -> three stored iterates
    alg.solve(
        np.array([2.0, 2.0]),
        max_iter=100,
        prox_tol=0.0,
        del_prox_tol=1e-8,
        del_prox_n=3,
        storage=True,
        storage_iters=2,
    )
    assert alg.all_x.shape == (3, 2)
    assert np.array_equal(alg.all_x[0], np.array([2.0, 2.0]))
    assert np.allclose(alg.all_x[1], np.array([1.0, 1.0]))  # iterate 2
    assert np.allclose(alg.all_x[2], np.array([1.0, 1.0]))  # iterate 4


def test_SequentialAMSHyperslab_solve_storage_iters_list(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    # only the requested iteration indices are kept (0 = initial point)
    alg.solve(
        np.array([2.0, 2.0]),
        max_iter=100,
        prox_tol=0.0,
        del_prox_tol=1e-8,
        del_prox_n=3,
        storage=True,
        storage_iters=[0, 2],
    )
    assert alg.all_x.shape == (2, 2)
    assert np.array_equal(alg.all_x[0], np.array([2.0, 2.0]))
    assert np.allclose(alg.all_x[1], np.array([1.0, 1.0]))


def test_SequentialAMSHyperslab_solve_proximity_measures(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    x_0 = np.array([2.0, 2.0])
    alg.solve(x_0.copy(), max_iter=100, proximity_measures=[("p_norm", 2), "max_norm"])
    # one row per iteration (incl. the initial point), one column per measure
    assert alg.proximities.shape == (2, 2)
    # the first row is the proximity of the starting point (cf. the
    # proximity test: p_norm 9/8, max_norm 2)
    assert np.allclose(alg.proximities[0], np.array([9 / 8, 2.0]))
    assert np.allclose(alg.proximities[-1], np.zeros(2))


def test_SequentialAMSHyperslab_solve_alternative_stopping_criterion(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    calls = {"n": 0}

    def stop_always(x, alg_):
        calls["n"] += 1
        return True

    # the alternative criterion replaces the proximity-based one and fires
    # after the first iteration
    alg.solve(np.array([2.0, 2.0]), max_iter=100, alternative_stopping_criterion=stop_always)
    assert calls["n"] == 1
    assert len(alg.proximities) == 2


def test_SequentialAMSHyperslab_solve_alternative_initial_call(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    x_0 = np.array([2.0, 2.0])
    # an initial call returning True stops the run before any projection
    x_n = alg.solve(
        x_0.copy(),
        max_iter=100,
        alternative_stopping_criterion_initial_call=lambda x, alg_: True,
    )
    assert np.array_equal(x_n, x_0)
    assert len(alg.proximities) == 1


def test_SequentialAMSHyperslab_solve_callback_halts(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    seen = []

    def callback(x, k):
        seen.append(k)
        return True

    # prox_tol=0 makes the proximity criterion unreachable; the callback's
    # truthy return stops the loop after the first iteration
    alg.solve(np.array([2.0, 2.0]), max_iter=100, prox_tol=0.0, callback=callback)
    assert seen == [1]
    assert len(alg.proximities) == 2


def test_SequentialAMSHyperslab_solve_max_iter_zero(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    x_0 = np.array([2.0, 2.0])
    x_n = alg.solve(x_0.copy(), max_iter=0, storage=True)
    # no iteration is performed: the start is returned unchanged and only its
    # proximity/storage entries exist
    assert np.array_equal(x_n, x_0)
    assert alg.proximities.shape == (1, 1)
    assert alg.all_x.shape == (1, 2)


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_alg",
    [
        lambda A, lb, ub: SequentialAMSHyperslab(A, lb, ub),
        lambda A, lb, ub: SequentialAMSHyperslab(A, lb, ub, algorithmic_relaxation=1.5),
        lambda A, lb, ub: SequentialAMSHyperslab(A, lb, ub, relaxation=0.5),
        lambda A, lb, ub: SequentialAMSHyperslab(A, lb, ub, cs=[3, 2, 1, 0]),
    ],
    ids=["default", "algorithmic_relaxation", "relaxation", "custom_cs"],
)
def test_SequentialAMSHyperslab_fejer_monotone(make_alg, get_full_variables):
    # sequential AMS projections (relaxation parameters in (0, 2)) are Fejér
    # monotone with respect to the feasible set; [0, 0] lies in its interior
    A, lb, ub = get_full_variables
    alg = make_alg(A, lb, ub)
    z = np.zeros(2)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = alg.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


def test_SequentialAMSHyperslab_project_idempotent_on_feasible(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    # a strictly interior point is a fixed point of the projection
    x = np.array([0.5, 0.5])
    assert np.array_equal(alg.project(x.copy()), x)


def test_SequentialAMSHyperslab_solve_reduces_proximity(
    get_SequentialAMSHyperslab_input_full,
):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    rng = np.random.default_rng(42)
    for _ in range(3):
        x = 4 * rng.standard_normal(2)
        alg.solve(x, max_iter=50)
        assert alg.proximities[-1][0] <= alg.proximities[0][0]


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SequentialAMSHyperslab_solve_int_input(get_SequentialAMSHyperslab_input_full):
    alg, _, _, _ = get_SequentialAMSHyperslab_input_full
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=100)
    assert np.issubdtype(x_n.dtype, np.floating)
    assert np.all(np.abs(x_n - np.array([1.0, 1.0])) < 1e-10)


def test_SequentialAMSHyperslab_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_full_variables
    alg_np = SequentialAMSHyperslab(A, lb, ub)
    x_np = alg_np.solve(np.array([2.0, 2.0]), max_iter=100, storage=True)

    alg_cp = SequentialAMSHyperslab(cp.asarray(A), cp.asarray(lb), cp.asarray(ub))
    x_cp = alg_cp.solve(cp.asarray([2.0, 2.0]), max_iter=100, storage=True)

    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
    # stored iterates are converted to host memory via .get()
    assert isinstance(alg_cp.all_x, np.ndarray)
    assert np.allclose(alg_cp.all_x, alg_np.all_x)


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------


def test_SequentialAMSHyperslab_superiorization_integration(get_full_variables):
    # the algorithm serves as the basic algorithm of a superiorization run
    # steered by a power-series gradient perturbation of x @ x
    A, lb, ub = get_full_variables
    basic = SequentialAMSHyperslab(A, lb, ub)
    pert = PowerSeriesGradientPerturbation(lambda x: x @ x, lambda x: 2 * x)
    sup = Superiorization(basic, pert)

    x = sup.solve(np.array([2.0, 2.0]), max_iter=10)

    assert x.shape == (2,)
    assert np.all(np.isfinite(x))
    assert pert._k > 0  # the driver advanced the perturbation each iteration
    # the result stays (near-)feasible while the objective was reduced
    assert basic.proximity(x, [("p_norm", 2)])[0] < 1e-6
    assert x @ x <= np.array([2.0, 2.0]) @ np.array([2.0, 2.0])
