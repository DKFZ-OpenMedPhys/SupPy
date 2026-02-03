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
    alg.all_function_values = [1, 2, 2]
    alg.proximities = [[0, 2], [10, 3], [0, 3]]
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
    alg.all_function_values = [1, 2, 2]
    alg.proximities = [[1, 2], [10, 3], [1, 3]]
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
    alg.all_function_values = [1, 10, 1]
    alg.proximities = [[1, 2], [10, 3], [0.8, 3]]
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
    alg.all_function_values = [1, 10, 1]
    alg.proximities = [[1, 2], [10, 3], [0, 3]]
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

    # success because of proximity value change below threshold
    alg = get_superiorization_input
    alg.all_function_values = [1, 10, 1]
    alg.proximities = [[1, 2], [10, 3], [1, 3]]
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

    # alg.f_k = 1
    # alg.p_k = 1
    # assert (
    #     alg._stopping_criteria(
    #         f_temp=1 + 9.9e-6, p_temp=[9.9e-6], objective_tol=1e-5, prox_tol=1e-5
    #     )
    #     == True
    # )


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
    x_1 = alg.solve(x, max_iter=1, storage=True)
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
    # this should be effectively two gradient steps only
