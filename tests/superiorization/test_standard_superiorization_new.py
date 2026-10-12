import pytest
import numpy as np
from suppy.feasibility import SequentialAMSHyperslab
from suppy.superiorization import Superiorization
from suppy.perturbations import PowerSeriesGradientPerturbation


@pytest.fixture
def get_full_variables():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    return A, lb, ub


@pytest.fixture
def get_SequentialAMSHyperslab_input_full(get_full_variables):
    A, lb, ub = get_full_variables
    return SequentialAMSHyperslab(A, lb, ub)


@pytest.fixture
def get_test_func():
    def func(x):
        return x @ x

    return func


@pytest.fixture
def get_test_grad():
    def grad(x):
        return 2 * x

    return grad


@pytest.fixture
def get_test_func_args():
    return [2.0]


@pytest.fixture
def get_test_grad_args():
    return [2.0]


@pytest.fixture
def get_test_perturbation(get_test_func, get_test_grad):
    return PowerSeriesGradientPerturbation(get_test_func, get_test_grad)


@pytest.fixture
def get_superiorization_input(get_SequentialAMSHyperslab_input_full, get_test_perturbation):
    return Superiorization(get_SequentialAMSHyperslab_input_full, get_test_perturbation)


def test_Superiorization_constructor(
    get_superiorization_input, get_SequentialAMSHyperslab_input_full, get_test_perturbation
):
    sup = get_superiorization_input

    assert sup.basic == get_SequentialAMSHyperslab_input_full
    assert sup.perturbation_scheme == get_test_perturbation

    assert sup._k == 0

    assert sup.all_x == []
    assert sup.all_function_values == []
    assert sup.proximities == []
    assert sup.all_x_function_reduction == []
    assert sup.all_function_values_function_reduction == []
    assert sup.proximities_function_reduction == []
    assert sup.all_x_basic == []
    assert sup.all_function_values_basic == []
    assert sup.proximities_basic == []


def test_Superiorization_stopping_criteria(get_superiorization_input):
    del_objective_tol = 1e-5
    del_objective_n = 1
    prox_tol = 1e-5
    del_prox_tol = 1e-6
    del_prox = 1

    # fail because of objective despite proximity below threshold
    alg = get_superiorization_input
    alg.all_function_values_basic = [1, 2]
    alg.proximities_basic = [[0, 2], [0, 3]]
    alg._n_tol_objective = (
        0  # number of iterations with objective function changes below threshold
    )
    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold

    assert (
        alg._stopping_criterion(
            del_objective_tol, del_objective_n, prox_tol, del_prox_tol, del_prox
        )
        == False
    )

    # fail because of objective but with correct proximity change
    alg = get_superiorization_input
    alg.all_function_values_basic = [1, 2]
    alg.proximities_basic = [[1, 2], [1, 3]]
    alg._n_tol_objective = (
        0  # number of iterations with objective function changes below threshold
    )
    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold
    assert (
        alg._stopping_criterion(
            del_objective_tol, del_objective_n, prox_tol, del_prox_tol, del_prox
        )
        == False
    )

    # fail because of proximity but with correct objective change
    alg = get_superiorization_input
    alg.all_function_values_basic = [1, 1]
    alg.proximities_basic = [[1, 2], [0.8, 3]]
    alg._n_tol_objective = (
        0  # number of iterations with objective function changes below threshold
    )
    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold
    assert (
        alg._stopping_criterion(
            del_objective_tol, del_objective_n, prox_tol, del_prox_tol, del_prox
        )
        == False
    )

    # success because of proximity values below threshold
    alg = get_superiorization_input
    alg.all_function_values_basic = [1, 1]
    alg.proximities_basic = [[1, 2], [0, 3]]
    alg._n_tol_objective = (
        0  # number of iterations with objective function changes below threshold
    )
    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold
    assert (
        alg._stopping_criterion(
            del_objective_tol, del_objective_n, prox_tol, del_prox_tol, del_prox
        )
        == True
    )

    # success because of proximity value CHANGE below threshold: the
    # proximity value itself stays above prox_tol, so only the del_prox
    # path can trigger the stop
    alg = get_superiorization_input
    alg.all_function_values_basic = [1, 1]
    alg.proximities_basic = [[1.0, 2], [1.0 - 1e-8, 3]]
    alg._n_tol_objective = (
        0  # number of iterations with objective function changes below threshold
    )
    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold
    assert (
        alg._stopping_criterion(
            del_objective_tol, del_objective_n, prox_tol, del_prox_tol, del_prox
        )
        == True
    )
    assert alg.proximities_basic[-1][0] > prox_tol  # the value-based path did not fire


def test_Superiorization_stopping_criteria_objective_increase(get_superiorization_input):
    """N consecutive objective increases stop the algorithm, independent of
    magnitude
    and independent of the small-decrease plateau count.
    """
    del_objective_tol = 1e-5
    del_objective_n = 5  # plateau path not exercised here
    del_objective_increase_n = 2
    prox_tol = 1e-5
    del_prox_tol = 1e-6
    del_prox_n = 1

    alg = get_superiorization_input
    alg.proximities_basic = [[0, 2], [0, 3]]  # proximity already satisfied throughout
    alg._n_tol_objective = 0
    alg._n_tol_increase = 0
    alg._n_tol_prox = 0

    # first increase (1 -> 2): not yet 2 in a row, so no stop
    alg.all_function_values_basic = [1, 2]
    assert (
        alg._stopping_criterion(
            del_objective_tol,
            del_objective_n,
            prox_tol,
            del_prox_tol,
            del_prox_n,
            del_objective_increase_n,
        )
        == False
    )
    assert alg._n_tol_increase == 1

    # second consecutive increase (2 -> 3): reaches del_objective_increase_n, stops
    alg.all_function_values_basic = [2, 3]
    assert (
        alg._stopping_criterion(
            del_objective_tol,
            del_objective_n,
            prox_tol,
            del_prox_tol,
            del_prox_n,
            del_objective_increase_n,
        )
        == True
    )
    assert alg._n_tol_increase == 2


def test_Superiorization_stopping_criteria_increase_and_plateau_reset_each_other(
    get_superiorization_input,
):
    """An objective increase resets the small-decrease plateau streak, and a
    subsequent non-increasing step resets the increase streak.
    """
    del_objective_tol = 1e-5
    del_objective_n = 1
    del_objective_increase_n = 5  # large enough to not trigger in this test
    prox_tol = 1e-5
    del_prox_tol = 1e-6
    del_prox_n = 1

    alg = get_superiorization_input
    # proximity value stays above prox_tol and its own change stays above del_prox_tol,
    # so stop_prox is False throughout and only the objective-side counters are exercised.
    alg.proximities_basic = [[1, 2], [0.8, 3]]
    alg._n_tol_objective = 0
    alg._n_tol_increase = 0
    alg._n_tol_prox = 0

    # unchanged objective (1 -> 1): counts as a (zero-magnitude) small decrease
    alg.all_function_values_basic = [1, 1]
    alg._stopping_criterion(
        del_objective_tol,
        del_objective_n,
        prox_tol,
        del_prox_tol,
        del_prox_n,
        del_objective_increase_n,
    )
    assert alg._n_tol_objective == 1
    assert alg._n_tol_increase == 0

    # an increase (1 -> 2) resets the plateau streak and starts the increase streak
    alg.all_function_values_basic = [1, 2]
    alg._stopping_criterion(
        del_objective_tol,
        del_objective_n,
        prox_tol,
        del_prox_tol,
        del_prox_n,
        del_objective_increase_n,
    )
    assert alg._n_tol_objective == 0
    assert alg._n_tol_increase == 1

    # a subsequent plateau step resets the increase streak back to 0
    alg.all_function_values_basic = [2, 2]
    alg._stopping_criterion(
        del_objective_tol,
        del_objective_n,
        prox_tol,
        del_prox_tol,
        del_prox_n,
        del_objective_increase_n,
    )
    assert alg._n_tol_increase == 0
    assert alg._n_tol_objective == 1


def test_initialize_storage(get_superiorization_input):
    alg = get_superiorization_input
    alg.all_x = [1, 2, 3]
    alg.all_function_values = [1, 2, 3]
    alg.all_x_function_reduction = [1, 2, 3]
    alg.all_function_values_function_reduction = [1, 2, 3]
    alg.all_x_basic = [1, 2, 3]
    alg.all_function_values_basic = [1, 2, 3]

    alg._initial_storage(np.array([1, 2]), True, 5, 4)
    assert np.array_equal(alg.all_x, [np.array([1, 2])])
    assert alg.all_function_values == [5]
    assert alg.proximities == [4]

    assert np.array_equal(alg.all_x_function_reduction, [np.array([1, 2])])
    assert alg.all_function_values_function_reduction == [5]
    assert alg.proximities_function_reduction == [4]

    assert np.array_equal(alg.all_x_basic, [np.array([1, 2])])
    assert alg.all_function_values_basic == [5]
    assert alg.proximities_basic == [4]


def test_storage_function_reduction(get_superiorization_input):
    alg = get_superiorization_input
    alg.storage(np.array([1, 2]), "function_reduction", True, 5, [4])
    assert np.array_equal(alg.all_x, [np.array([1, 2])])
    assert alg.all_function_values == [5]
    assert alg.proximities == [[4]]

    assert np.array_equal(alg.all_x_function_reduction, [np.array([1, 2])])
    assert alg.all_function_values_function_reduction == [5]
    assert alg.proximities_function_reduction == [[4]]

    assert alg.all_x_basic == []
    assert alg.all_function_values_basic == []
    assert alg.proximities_basic == []


def test_storage_basic_step(get_superiorization_input):
    alg = get_superiorization_input
    alg.storage(np.array([1, 2]), "basic", True, 5, [4])
    assert np.array_equal(alg.all_x, [np.array([1, 2])])
    assert alg.all_function_values == [5]
    assert alg.proximities == [[4]]

    assert np.array_equal(alg.all_x_basic, [np.array([1, 2])])
    assert alg.all_function_values_basic == [5]
    assert alg.proximities_basic == [[4]]

    assert alg.all_x_function_reduction == []
    assert alg.all_function_values_function_reduction == []
    assert alg.proximities_function_reduction == []


def test_PowerSeriesGradient_superiorization_step_only_one_function_reduction(
    get_superiorization_input,
):
    alg = get_superiorization_input
    x = np.array([1, 1])
    # perform a single iteration
    x_1 = alg.solve(x, max_iter=1)
    # this should be effectively only a single gradient step
    assert np.array_equal(x_1, (1 - 1 / np.sqrt(2)) * np.array([1, 1]))


def test_PowerSeriesGradient_superiorization_step_only_two_function_reduction(
    get_superiorization_input,
):
    alg = get_superiorization_input
    alg.perturbation_scheme.n_red = 2
    x = np.array([1, 1])
    # perform a single iteration
    x_1 = alg.solve(x, max_iter=1)
    # this should be effectively two gradient steps only
    assert np.array_equal(x_1, (1 - 1 / np.sqrt(2) - 1 / np.sqrt(8)) * np.array([1, 1]))


def test_PowerSeriesGradient_superiorization(get_superiorization_input):
    alg = get_superiorization_input
    x = np.array([2, 2])
    # perform a single iteration
    x_1 = alg.solve(x, max_iter=1, storage=True, store_function_reduction=True)
    # this should be one gradient and one basic step
    assert np.array_equal(x_1, np.array([1.0, 1.0]))
    assert np.array_equal(
        alg.all_x,
        [np.array([2, 2]), (2 - 1 / np.sqrt(2)) * np.array([1, 1]), np.array([1, 1])],
    )
    assert np.array_equal(
        alg.all_x_function_reduction, [np.array([2, 2]), (2 - 1 / np.sqrt(2)) * np.array([1, 1])]
    )
    assert np.all(abs(alg.all_function_values - np.array([8, 9.0 - 4.0 * np.sqrt(2), 2])) < 1e-10)
    assert np.all(
        alg.all_function_values_function_reduction - np.array([8, 9 - 4 * np.sqrt(2)]) < 1e-10
    )
    assert np.array_equal(alg.all_x_basic, [np.array([2, 2]), np.array([1, 1])])
    assert np.array_equal(alg.all_function_values_basic, [8, 2])


# ---------------------------------------------------------------------------
# Superiorization.solve: options and branches
# ---------------------------------------------------------------------------
# Reference trajectory from x0 = [2, 2] (PowerSeries default, func = x @ x):
# iter 1: function reduction -> (2 - 1/sqrt(2))*[1, 1], basic -> [1, 1]
# iter 2: function reduction -> (1 - 1/sqrt(8))*[1, 1], basic step is a no-op
# (the iterate is already feasible)


def test_Superiorization_solve_storage_iters_int(get_superiorization_input):
    alg = get_superiorization_input
    # storage_iters=2 stores index 0 (initial) and index 2 (both snapshots
    # of iteration 2); iteration 1 (index 1) is skipped
    alg.solve(
        np.array([2.0, 2.0]),
        max_iter=2,
        storage=True,
        store_function_reduction=True,
        storage_iters=2,
    )
    c2 = 1 - 1 / np.sqrt(8)
    assert alg.all_x.shape == (3, 2)
    assert np.allclose(alg.all_x[0], np.array([2.0, 2.0]))
    assert np.allclose(alg.all_x[1], c2 * np.ones(2))  # function reduction of iter 2
    assert np.allclose(alg.all_x[2], c2 * np.ones(2))  # basic of iter 2 (no-op)
    # function values are always recorded regardless of storage_iters
    assert len(alg.all_function_values) == 5


def test_Superiorization_solve_storage_iters_list(get_superiorization_input):
    alg = get_superiorization_input
    # the list form stores exactly the given indices: 0 (initial) and 1
    alg.solve(
        np.array([2.0, 2.0]),
        max_iter=2,
        storage=True,
        store_function_reduction=True,
        storage_iters=[0, 1],
    )
    assert alg.all_x.shape == (3, 2)
    assert np.allclose(alg.all_x[0], np.array([2.0, 2.0]))
    assert np.allclose(alg.all_x[1], (2 - 1 / np.sqrt(2)) * np.ones(2))
    assert np.allclose(alg.all_x[2], np.array([1.0, 1.0]))


def test_Superiorization_solve_no_function_reduction_storage(get_superiorization_input):
    alg = get_superiorization_input
    # store_function_reduction defaults to False: the function-reduction
    # snapshot is skipped entirely; only initial + basic values are recorded
    alg.solve(np.array([2.0, 2.0]), max_iter=1, storage=True)
    assert alg.all_x.shape == (2, 2)  # initial + basic iterate only
    assert np.allclose(alg.all_x[1], np.array([1.0, 1.0]))
    assert np.allclose(alg.all_function_values, np.array([8.0, 2.0]))
    # the function-reduction histories keep only their initial entries
    assert len(alg.all_function_values_function_reduction) == 1
    assert len(alg.proximities_function_reduction) == 1
    assert alg.all_x_function_reduction.shape == (1, 2)


def test_Superiorization_solve_max_iter_zero(get_superiorization_input):
    alg = get_superiorization_input
    x_n = alg.solve(np.array([2.0, 2.0]), max_iter=0, storage=True)
    assert np.array_equal(x_n, np.array([2.0, 2.0]))
    assert alg._k == 0
    assert len(alg.all_function_values) == 1
    assert len(alg.proximities_basic) == 1


def test_Superiorization_solve_multiple_proximity_measures(get_superiorization_input):
    alg = get_superiorization_input
    alg.solve(
        np.array([2.0, 2.0]),
        max_iter=1,
        proximity_measures=[("p_norm", 2), "max_norm"],
    )
    # one entry per recorded snapshot, one column per measure
    assert alg.proximities.shape == (2, 2)


def test_Superiorization_storage_invalid_kind(get_superiorization_input):
    alg = get_superiorization_input
    with pytest.raises(ValueError):
        alg.storage(np.array([1.0, 2.0]), kind="bogus", storage=True, f=1.0, p=[0.0])


def test_Superiorization_alternative_stopping_criterion(get_superiorization_input):
    alg = get_superiorization_input
    alg.solve(
        np.array([2.0, 2.0]),
        max_iter=100,
        alternative_stopping_criterion=lambda x, sup: sup._k >= 2,
    )
    assert alg._k == 2

    # the initial call can stop the solve before any iteration
    alg = get_superiorization_input
    x_n = alg.solve(
        np.array([2.0, 2.0]),
        max_iter=100,
        alternative_stopping_criterion_initial_call=lambda x, sup: True,
    )
    assert alg._k == 0
    assert np.array_equal(x_n, np.array([2.0, 2.0]))


def test_Superiorization_solve_stops_early_via_public_api(get_superiorization_input):
    # feasible problem: proximity reaches 0 after one iteration and the
    # perturbation steps shrink geometrically, so the combined criterion
    # fires long before max_iter
    alg = get_superiorization_input
    alg.solve(np.array([2.0, 2.0]), max_iter=100)
    assert alg._k < 100
    assert alg.proximities_basic[-1][0] < 1e-6


def test_Superiorization_solve_stall_exit_on_infeasible(get_test_func, get_test_grad):
    # infeasible system: the same row must lie in two disjoint bands, the
    # basic step reaches a fixed point with proximity above prox_tol and the
    # del_prox change criterion triggers the stop
    A = np.array([[1, 0], [1, 0]])
    lb = np.array([0.0, 1.0])
    ub = np.array([0.5, 1.5])
    basic = SequentialAMSHyperslab(A, lb, ub)
    pert = PowerSeriesGradientPerturbation(get_test_func, get_test_grad)
    alg = Superiorization(basic, pert)

    alg.solve(np.array([2.0, 2.0]), max_iter=50, del_objective_n=2, del_prox_n=2)
    assert alg._k < 50
    assert alg.proximities_basic[-1][0] > 1e-6  # still infeasible at the stop


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_Superiorization_resolve_reproducible(get_superiorization_input):
    # solve() resets the perturbation scheme, so running twice from the same
    # start must reproduce the identical trajectory
    alg = get_superiorization_input
    alg.solve(np.array([2.0, 2.0]), max_iter=3, storage=True, store_function_reduction=True)
    first_all_x = alg.all_x.copy()
    first_f = alg.all_function_values.copy()

    alg.solve(np.array([2.0, 2.0]), max_iter=3, storage=True, store_function_reduction=True)
    assert np.allclose(alg.all_x, first_all_x)
    assert np.allclose(alg.all_function_values, first_f)


def test_Superiorization_timing_reset_between_solves(get_superiorization_input):
    # t/t_it are per-solve records: a second solve starts fresh instead of
    # accumulating mixed-epoch timings
    alg = get_superiorization_input
    alg.solve(np.array([2.0, 2.0]), max_iter=3)
    alg.solve(np.array([2.0, 2.0]), max_iter=3)
    assert alg.t[0] == 0
    assert len(alg.t) == alg._k + 1
    assert len(alg.t_it) == alg._k


def test_Superiorization_history_shape_invariant(get_superiorization_input):
    # with function-reduction storage on, each iteration adds two combined
    # snapshots and one snapshot per sub-history
    alg = get_superiorization_input
    alg.solve(np.array([2.0, 2.0]), max_iter=3, storage=True, store_function_reduction=True)
    k = alg._k
    assert len(alg.all_function_values) == 2 * k + 1
    assert len(alg.all_function_values_basic) == k + 1
    assert len(alg.all_function_values_function_reduction) == k + 1
    assert alg.all_x.shape == (2 * k + 1, 2)
    assert len(alg.proximities_basic) == k + 1


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_Superiorization_solve_int_input(get_superiorization_input):
    alg = get_superiorization_input
    # @ensure_float_array coerces the integer starting point to float64
    x_n = alg.solve(np.array([2, 2]), max_iter=1)
    assert np.issubdtype(x_n.dtype, np.floating)


def test_Superiorization_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_full_variables

    def make_sup(basic):
        # backend-agnostic objective (x @ x works on numpy and cupy)
        pert = PowerSeriesGradientPerturbation(lambda x: x @ x, lambda x: 2 * x)
        return Superiorization(basic, pert)

    sup_np = make_sup(SequentialAMSHyperslab(A, lb, ub))
    x_np = sup_np.solve(
        np.array([2.0, 2.0]), max_iter=2, storage=True, store_function_reduction=True
    )

    sup_cp = make_sup(SequentialAMSHyperslab(cp.asarray(A), cp.asarray(lb), cp.asarray(ub)))
    x_cp = sup_cp.solve(
        cp.asarray([2.0, 2.0]), max_iter=2, storage=True, store_function_reduction=True
    )

    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
    # _post_step converts every history to numpy via .get()
    assert isinstance(sup_cp.all_x, np.ndarray)
    assert np.allclose(sup_cp.all_x, sup_np.all_x)
    assert np.allclose(sup_cp.all_function_values, sup_np.all_function_values)
    assert np.allclose(sup_cp.proximities, sup_np.proximities)
