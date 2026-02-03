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


def test_Superiorization_constructor(
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


def test_Superiorization_stopping_criteria(get_superiorization_input):
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
