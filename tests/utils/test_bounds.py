import numpy as np
from suppy.utils import Bounds
import pytest


def test_bounds():
    lb = np.array([1, 2, 3])
    ub = np.array([4, 5, 6])

    bounds = Bounds(lb, ub)
    bounds_2 = Bounds(lb=lb)
    bounds_3 = Bounds(ub=ub)
    with pytest.raises(ValueError):
        bounds_4 = Bounds()

    # check first bounds object
    assert np.array_equal(bounds.l, lb)
    assert np.array_equal(bounds.u, ub)

    x = np.array([2, 3, 4])

    r1, r2 = bounds.residual(x)
    assert np.array_equal(r1, x - lb)
    assert np.array_equal(r2, ub - x)

    i = 1
    r1_i, r2_i = bounds.single_residual(x[i], i)
    assert r1_i == x[i] - lb[i]
    assert r2_i == ub[i] - x[i]

    assert np.array_equal(bounds._center(), (lb + ub) / 2)

    assert np.array_equal(bounds._half_distance(), (ub - lb) / 2)
