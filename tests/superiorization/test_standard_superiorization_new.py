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

    assert sup.f_k == None
    assert sup.p_k == None
    assert sup._k == 0

    assert sup.all_x == []
    assert sup.all_function_values == []
    assert sup.all_proximity_values == []
    assert sup.all_x_function_reduction == []
    assert sup.all_function_values_function_reduction == []
    assert sup.all_proximity_values_function_reduction == []
    assert sup.all_x_basic == []
    assert sup.all_function_values_basic == []
    assert sup.all_proximity_values_basic == []


def test_Superiorization_stopping_criteria(get_superiorization_input):
    alg = get_superiorization_input
    alg.f_k = 1
    alg.p_k = 2
    assert alg._stopping_criteria(f_temp=2, p_temp=3, objective_tol=1e-5, constr_tol=1e-5) == False

    alg.f_k = 1
    alg.p_k = 1
    assert (
        alg._stopping_criteria(
            f_temp=1 + 9.9e-6, p_temp=[9.9e-6], objective_tol=1e-5, constr_tol=1e-5
        )
        == True
    )


def test_initialize_storage(get_superiorization_input):
    alg = get_superiorization_input
    alg.all_x = [1, 2, 3]
    alg.all_function_values = [1, 2, 3]
    alg.all_x_function_reduction = [1, 2, 3]
    alg.all_function_values_function_reduction = [1, 2, 3]
    alg.all_x_basic = [1, 2, 3]
    alg.all_function_values_basic = [1, 2, 3]

    alg._initial_storage(np.array([1, 2]), 5, 4)
    assert np.array_equal(alg.all_x, [np.array([1, 2])])
    assert alg.all_function_values == [5]
    assert alg.all_proximity_values == [4]

    assert alg.all_x_function_reduction == []
    assert alg.all_function_values_function_reduction == []
    assert alg.all_proximity_values_function_reduction == []

    assert alg.all_x_basic == []
    assert alg.all_function_values_basic == []
    assert alg.all_proximity_values_basic == []


def test_storage_function_reduction(get_superiorization_input):
    alg = get_superiorization_input
    alg._storage_function_reduction(np.array([1, 2]), 5, [4])
    assert np.array_equal(alg.all_x, [np.array([1, 2])])
    assert alg.all_function_values == [5]
    assert alg.all_proximity_values == [[4]]

    assert np.array_equal(alg.all_x_function_reduction, [np.array([1, 2])])
    assert alg.all_function_values_function_reduction == [5]
    assert alg.all_proximity_values_function_reduction == [[4]]

    assert alg.all_x_basic == []
    assert alg.all_function_values_basic == []
    assert alg.all_proximity_values_basic == []


def test_storage_basic_step(get_superiorization_input):
    alg = get_superiorization_input
    alg._storage_basic_step(np.array([1, 2]), 5, [4])
    assert np.array_equal(alg.all_x, [np.array([1, 2])])
    assert alg.all_function_values == [5]
    assert alg.all_proximity_values == [[4]]

    assert np.array_equal(alg.all_x_basic, [np.array([1, 2])])
    assert alg.all_function_values_basic == [5]
    assert alg.all_proximity_values_basic == [[4]]

    assert alg.all_x_function_reduction == []
    assert alg.all_function_values_function_reduction == []
    assert alg.all_proximity_values_function_reduction == []


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
    assert np.array_equal(alg.all_x_function_reduction, [(2 - 1 / np.sqrt(2)) * np.array([1, 1])])
    assert np.all(
        np.abs(alg.all_function_values - np.array([8, 9.0 - 4.0 * np.sqrt(2), 2])) < 1e-10
    )
    assert np.all(
        alg.all_function_values_function_reduction - np.array([9 - 4 * np.sqrt(2)]) < 1e-10
    )
    assert np.array_equal(alg.all_x_basic, [np.array([1, 1])])
    assert np.array_equal(alg.all_function_values_basic, [2])
