import numpy as np
import pytest
from suppy.projections import CustomProjection


def _project(x):
    return np.maximum(x, 0)


def test_custom_projection():
    custom_proj = CustomProjection(projection_function=_project)

    x = np.array([-1.0, 2.0, -3.0, 4.0])
    projected_x = custom_proj.project(x)
    expected_projection = np.array([0.0, 2.0, 0.0, 4.0])
    assert np.array_equal(
        projected_x, expected_projection
    ), "Custom projection did not work as expected."

    proximities = custom_proj.proximity(x, proximity_measures=[("p_norm", 2)])
    expected_proximity = [np.linalg.norm(expected_projection - x, 2) ** 2]
    assert np.allclose(
        proximities, expected_proximity
    ), "Custom proximity calculation did not work as expected."
