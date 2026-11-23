import pytest
import numpy as np
from suppy.feasibility import SequentialAMSHyperslab
from suppy.projections import BoxProjection
from suppy.superiorization import SplitSuperiorization
from suppy.perturbations import PowerSeriesGradientPerturbation, DummyPerturbation
from suppy.feasibility import CQAlgorithm


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
def get_basic_algorithm_input(get_full_variables):
    A, lb, ub = get_full_variables
    q_projection = BoxProjection(lb, ub)
    c_projection = BoxProjection(0, np.inf)
    split = CQAlgorithm(A, c_projection, q_projection)
    return split


@pytest.fixture
def get_test_perturbation(get_test_func, get_test_grad):
    return PowerSeriesGradientPerturbation(get_test_func, get_test_grad)


@pytest.fixture
def get_superiorization_input(get_basic_algorithm_input, get_test_perturbation):
    return SplitSuperiorization(get_basic_algorithm_input, get_test_perturbation)


def test_SplitSuperiorization_constructor(
    get_superiorization_input, get_basic_algorithm_input, get_test_perturbation
):
    sup = get_superiorization_input

    assert sup.basic == get_basic_algorithm_input
    assert sup.input_perturbation_scheme == get_test_perturbation
    assert isinstance(sup.target_perturbation_scheme, DummyPerturbation)

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


def test_SplitSuperiorization_stopping_criteria(get_superiorization_input):
    del_input_objective_tol = 1e-5
    del_target_objective_tol = 1e-5
    del_input_objective_n = 1
    del_target_objective_n = 1
    prox_tol = 1e-5
    del_prox_tol = 1e-6
    del_prox_n = 1

    # fail because of objective despite proximity below threshold
    alg = get_superiorization_input
    alg.all_function_values = [[0, 1], [0, 2], [0, 2]]
    alg.proximities = [[[0, 0], [0, 2]], [[0, 10], [0, 3]], [[0, 0], [0, 3]]]
    alg._n_tol_input_objective = (
        0  # number of iterations with input objective function changes below threshold
    )
    alg._n_tol_target_objective = (
        0  # number of iterations with target objective function changes below threshold
    )

    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold

    assert (
        alg._stopping_criterion(
            del_input_objective_tol=del_input_objective_tol,
            del_input_objective_n=del_input_objective_n,
            del_target_objective_tol=del_target_objective_tol,
            del_target_objective_n=del_target_objective_n,
            prox_tol=prox_tol,
            del_prox_tol=del_prox_tol,
            del_prox_n=del_prox_n,
        )
        == False
    )

    alg = get_superiorization_input
    alg.all_function_values = [[0, 1], [0, 2], [0, 2]]
    alg.proximities = [[[0, 1], [0, 2]], [[0, 10], [0, 3]], [[1, 0], [3, 0]]]
    alg._n_tol_input_objective = (
        0  # number of iterations with objective function changes below threshold
    )
    alg._n_tol_target_objective = (
        0  # number of iterations with target objective function changes below threshold
    )

    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold

    assert (
        alg._stopping_criterion(
            del_input_objective_tol=del_input_objective_tol,
            del_input_objective_n=del_input_objective_n,
            del_target_objective_tol=del_target_objective_tol,
            del_target_objective_n=del_target_objective_n,
            prox_tol=prox_tol,
            del_prox_tol=del_prox_tol,
            del_prox_n=del_prox_n,
        )
        == False
    )

    # success because of proximity values below threshold

    alg = get_superiorization_input
    alg.all_function_values = [[0, 1], [0, 10], [0, 1]]
    alg.proximities = [[[0, 1], [0, 2]], [[0, 10], [0, 3]], [[0, 0], [0, 3]]]
    alg._n_tol_input_objective = (
        0  # number of iterations with input objective function changes below threshold
    )
    alg._n_tol_target_objective = (
        0  # number of iterations with target objective function changes below threshold
    )

    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold

    assert (
        alg._stopping_criterion(
            del_input_objective_tol=del_input_objective_tol,
            del_input_objective_n=del_input_objective_n,
            del_target_objective_tol=del_target_objective_tol,
            del_target_objective_n=del_target_objective_n,
            prox_tol=prox_tol,
            del_prox_tol=del_prox_tol,
            del_prox_n=del_prox_n,
        )
        == True
    )

    # success because of proximity value CHANGE below threshold: the
    # target-space proximity value itself stays above prox_tol, so only the
    # del_prox path can trigger the stop
    alg = get_superiorization_input
    alg.all_function_values = [[0, 1], [0, 10], [0, 1]]
    alg.proximities = [[[0, 1], [1.0, 2]], [[0, 10], [0.5, 3]], [[0, 0], [1.0 - 1e-8, 3]]]
    alg._n_tol_input_objective = (
        0  # number of iterations with input objective function changes below threshold
    )
    alg._n_tol_target_objective = (
        0  # number of iterations with target objective function changes below threshold
    )

    alg._n_tol_prox = 0  # number of iterations with proximity changes below threshold

    assert (
        alg._stopping_criterion(
            del_input_objective_tol=del_input_objective_tol,
            del_input_objective_n=del_input_objective_n,
            del_target_objective_tol=del_target_objective_tol,
            del_target_objective_n=del_target_objective_n,
            prox_tol=prox_tol,
            del_prox_tol=del_prox_tol,
            del_prox_n=del_prox_n,
        )
        == True
    )
    assert alg.proximities[-1][1][0] > prox_tol  # the value-based path did not fire


def test_split_superiorization_solve(get_superiorization_input):
    alg = get_superiorization_input

    # check initial storage
    alg.solve(np.array([2.0, 2.0]), max_iter=0, storage=True)
    assert np.array_equal(alg.all_x, np.array([np.array([2.0, 2.0])]))
    assert np.array_equal(alg.all_function_values, np.array([[8.0, 0]]))
    assert np.array_equal(alg.proximities, np.array([[[0], [1.125]]]))

    assert np.array_equal(alg.all_x_function_reduction, np.array([np.array([2.0, 2.0])]))
    assert np.array_equal(alg.all_function_values_function_reduction, np.array([[8.0, 0]]))
    assert np.array_equal(alg.proximities_function_reduction, np.array([[[0], [1.125]]]))

    assert np.array_equal(alg.all_x_basic, np.array([np.array([2.0, 2.0])]))
    assert np.array_equal(alg.all_function_values_basic, np.array([[8.0, 0]]))
    assert np.array_equal(alg.proximities_basic, np.array([[[0], [1.125]]]))

    alg = get_superiorization_input
    alg.solve(np.array([2.0, 2.0]), max_iter=1, storage=True)
    assert np.array_equal(
        alg.all_x,
        np.array(
            [
                np.array([2.0, 2.0]),
                (2 - 1 / np.sqrt(2)) * np.array([1, 1]),
                1 / np.sqrt(2) * np.array([1.0, 1.0]),
            ]
        ),
    )
    np.testing.assert_allclose(
        alg.all_function_values,
        np.array([[8.0, 0], [(2 - 1 / np.sqrt(2)) ** 2 * 2, 0], [1.0, 0.0]]),
    )
    np.testing.assert_allclose(
        alg.proximities,
        np.array([[[0], [1.125]], [[0], [1 / 4 * (2 - 2 * np.sqrt(1 / 2)) ** 2]], [[0], [0]]]),
    )

    assert np.array_equal(
        alg.all_x_basic, np.array([np.array([2.0, 2.0]), 1 / np.sqrt(2) * np.array([1.0, 1.0])])
    )
    np.testing.assert_allclose(alg.all_function_values_basic, np.array([[8.0, 0], [1.0, 0.0]]))
    np.testing.assert_allclose(alg.proximities_basic, np.array([[[0], [1.125]], [[0], [0]]]))

    assert np.array_equal(
        alg.all_x_function_reduction,
        np.array([np.array([2.0, 2.0]), (2 - 1 / np.sqrt(2)) * np.array([1, 1])]),
    )
    np.testing.assert_allclose(
        alg.all_function_values_function_reduction,
        np.array([[8.0, 0], [(2 - 1 / np.sqrt(2)) ** 2 * 2, 0]]),
    )
    np.testing.assert_allclose(
        alg.proximities_function_reduction,
        np.array([[[0], [1.125]], [[0], [1 / 4 * (2 - 2 * np.sqrt(1 / 2)) ** 2]]]),
    )


# ---------------------------------------------------------------------------
# SplitSuperiorization: options and branches
# ---------------------------------------------------------------------------


def test_SplitSuperiorization_constructor_requires_scheme(get_basic_algorithm_input):
    # the constructor's only validation: at least one perturbation scheme
    with pytest.raises(ValueError):
        SplitSuperiorization(get_basic_algorithm_input)


def test_SplitSuperiorization_constructor_target_only(
    get_basic_algorithm_input, get_test_perturbation
):
    # passing only a target scheme defaults the input scheme to a dummy
    sup = SplitSuperiorization(
        get_basic_algorithm_input, target_perturbation_scheme=get_test_perturbation
    )
    assert isinstance(sup.input_perturbation_scheme, DummyPerturbation)
    assert sup.target_perturbation_scheme == get_test_perturbation


def test_SplitSuperiorization_solve_storage_iters_int(get_superiorization_input):
    alg = get_superiorization_input
    # storage_iters=2 with max_iter=1 stores only index 0; function and
    # proximity values are always recorded regardless
    alg.solve(np.array([2.0, 2.0]), max_iter=1, storage=True, storage_iters=2)
    assert alg.all_x.shape == (1, 2)
    assert np.allclose(alg.all_x[0], np.array([2.0, 2.0]))
    assert len(alg.all_function_values) == 3  # initial + function reduction + basic


def test_SplitSuperiorization_storage_invalid_kind(get_superiorization_input):
    alg = get_superiorization_input
    with pytest.raises(ValueError):
        alg.storage(
            np.array([1.0, 2.0]), kind="bogus", storage=True, f=[1.0, 0.0], p=[[0.0], [0.0]]
        )


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_SplitSuperiorization_solve_stops_early_via_public_api(get_superiorization_input):
    # the CQ problem is feasible (target proximity reaches 0) and the
    # perturbation steps shrink geometrically, so with n=1 the combined
    # criterion fires long before max_iter
    alg = get_superiorization_input
    alg.solve(
        np.array([2.0, 2.0]),
        max_iter=50,
        del_input_objective_n=1,
        del_target_objective_n=1,
        del_prox_n=1,
    )
    assert alg._k < 50
    assert alg.proximities[-1][1][0] < 1e-6


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_SplitSuperiorization_cupy_backend(get_full_variables):
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    A, lb, ub = get_full_variables

    def make_sup(A_backend, lb_backend, ub_backend):
        q_projection = BoxProjection(lb_backend, ub_backend)
        c_projection = BoxProjection(0, np.inf)
        # backend-agnostic objective (x @ x works on numpy and cupy)
        pert = PowerSeriesGradientPerturbation(lambda x: x @ x, lambda x: 2 * x)
        return SplitSuperiorization(CQAlgorithm(A_backend, c_projection, q_projection), pert)

    sup_np = make_sup(np.asarray(A, dtype=float), lb, ub)
    x_np = sup_np.solve(np.array([2.0, 2.0]), max_iter=1, storage=True)

    sup_cp = make_sup(cp.asarray(A, dtype=float), cp.asarray(lb), cp.asarray(ub))
    x_cp = sup_cp.solve(cp.asarray([2.0, 2.0]), max_iter=1, storage=True)

    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
    # storage/_post_step convert the histories to numpy via .get()
    assert isinstance(sup_cp.all_x, np.ndarray)
    assert np.allclose(sup_cp.all_x, sup_np.all_x)
    assert np.allclose(sup_cp.all_function_values, sup_np.all_function_values)
    assert np.allclose(sup_cp.proximities, sup_np.proximities)
