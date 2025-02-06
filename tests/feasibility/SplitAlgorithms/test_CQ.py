import numpy as np
from suppy.feasibility import CQAlgorithm
from suppy.projections import BoxProjection
from suppy.utils import LinearMapping
import pytest


@pytest.fixture
def get_simple_cq():
    A = np.array([[0, 1], [-1, 0]])
    c_projection = BoxProjection(0, 1)
    q_projection = BoxProjection(0, 2)
    return CQAlgorithm(A, c_projection, q_projection), A, c_projection, q_projection


def test_cq_constructor(get_simple_cq):
    cq, _, _, _ = get_simple_cq
    assert isinstance(cq, CQAlgorithm)
    assert isinstance(cq.A, LinearMapping)
    assert isinstance(cq.c_projection, BoxProjection)
    assert isinstance(cq.q_projection, BoxProjection)
    assert cq.relaxation == 1
    assert cq.algorithmic_relaxation == 1
    assert cq.proximities == []


def test_cq_map(get_simple_cq):
    cq, _, _, _ = get_simple_cq
    x = np.array([2, 3])
    assert np.array_equal(cq.map(x), np.array([3, -2]))


def test_cq_map_back(get_simple_cq):
    cq, _, _, _ = get_simple_cq
    y = np.array([2, 3])
    assert np.array_equal(cq.map_back(y), np.array([-3, 2]))


def test_cq_proximity(get_simple_cq):
    cq, _, _, _ = get_simple_cq
    proximity_measures = [("p_norm", 2), "max_norm"]
    x = np.array([3, 3])
    assert cq.proximity(x, proximity_measures)[0] == 5.0
    assert cq.proximity(x, proximity_measures)[1] == 3.0


def test_cq_step(get_simple_cq):
    cq, _, _, _ = get_simple_cq
    x_1 = np.array([1, 1])
    x_2 = np.array([3, 3])
    s_1_x, s_1_y = cq.step(x_1)
    s_2_x, s_2_y = cq.step(x_2)
    assert np.array_equal(s_1_x, np.array([0, 1]))
    assert np.array_equal(s_1_y, np.array([1, 0]))
    assert np.array_equal(s_2_x, np.array([0, 1]))
    assert np.array_equal(s_2_y, np.array([2, 0]))


def test_cq_relaxed_step(get_simple_cq):
    cq, A, c, q = get_simple_cq
    cq_relaxed = CQAlgorithm(A, c, q, algorithmic_relaxation=2.0)
    x_1 = np.array([1, 1])
    x_2 = np.array([3, 3])
    s_1_x, s_1_y = cq_relaxed.step(x_1)
    s_2_x, s_2_y = cq_relaxed.step(x_2)
    assert np.array_equal(s_1_x, np.array([0.0, 1.0]))
    assert np.array_equal(s_1_y, np.array([1.0, 0.0]))
    assert np.array_equal(s_2_x, np.array([0.0, 1.0]))
    assert np.array_equal(s_2_y, np.array([2.0, 0.0]))


def test_cq_solve(get_simple_cq):
    cq, _, _, _ = get_simple_cq
    x = np.array([3, 3])
    x_n = cq.solve(x)
    assert np.array_equal(x_n, np.array([0, 1]))
    assert np.array_equal(cq.proximities, [[5.0], [0.0]])
