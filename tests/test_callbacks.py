import numpy as np
import pytest
from suppy.feasibility import SequentialAMSHyperslab
from suppy.superiorization import Superiorization
from suppy.perturbations import PowerSeriesGradientPerturbation


@pytest.fixture
def get_feasibility():
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    return SequentialAMSHyperslab(A, lb, ub)


@pytest.fixture
def get_infeasible():
    # first coordinate must lie in [2, 3] and simultaneously in [-3, -2]:
    # no common point, so the algorithm never reaches feasibility and keeps
    # iterating until max_iter (or the callback halts it).
    A = np.array([[1, 0], [1, 0]])
    lb = np.array([2, -3])
    ub = np.array([3, -2])
    return SequentialAMSHyperslab(A, lb, ub)


@pytest.fixture
def get_superiorization(get_feasibility):
    def func(x):
        return x @ x

    def grad(x):
        return 2 * x

    perturbation = PowerSeriesGradientPerturbation(func, grad)
    return Superiorization(get_feasibility, perturbation)


def test_feasibility_callback_observation(get_feasibility):
    alg = get_feasibility
    x = np.array([5.0, 5.0])
    recorded = []
    alg.solve(x, max_iter=10, callback=lambda x, k: recorded.append((k, float(x @ x))))

    # one callback per performed iteration, indices start at 1 and increase
    ks = [k for k, _ in recorded]
    assert ks == list(range(1, len(recorded) + 1))
    # number of stored proximities is initial point + one per iteration
    assert len(alg.proximities) == len(recorded) + 1


def test_feasibility_callback_halt(get_infeasible):
    alg = get_infeasible
    x = np.array([0.0, 0.0])
    recorded = []

    def callback(x, k):
        recorded.append(k)
        return k >= 3

    alg.solve(x, max_iter=500, callback=callback)

    # halts after exactly 3 iterations -> initial point + 3 proximities
    assert recorded == [1, 2, 3]
    assert len(alg.proximities) == 4


def test_superiorization_callback_observation(get_superiorization):
    sup = get_superiorization
    x = np.array([5.0, 5.0])
    recorded = []
    sup.solve(x, max_iter=10, callback=lambda x, k: recorded.append((k, float(x @ x))))

    ks = [k for k, _ in recorded]
    assert ks == list(range(1, len(recorded) + 1))


def test_superiorization_callback_halt(get_superiorization):
    sup = get_superiorization
    x = np.array([5.0, 5.0])
    sup.solve(x, max_iter=500, callback=lambda x, k: k >= 3)

    assert sup._k == 3
