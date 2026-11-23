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


# ---------------------------------------------------------------------------
# CustomProjection: options and branches
# ---------------------------------------------------------------------------


def test_custom_projection_custom_proximity_function():
    # a user-supplied proximity function replaces the default residual-based
    # calculation
    custom_proj = CustomProjection(
        projection_function=_project,
        proximity_function=lambda x, proximity_measures: [42.0],
    )
    prox = custom_proj.proximity(np.array([-1.0, 2.0]), [("p_norm", 2)])
    assert np.allclose(prox, [42.0])


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------


def test_custom_projection_in_sequential_solve():
    # a user-defined projection composes with library projections inside a
    # combination method and its solve loop (the extension contract)
    from suppy.projections import BoxProjection, SequentialProjection

    custom_proj = CustomProjection(projection_function=_project)  # x >= 0
    box_proj = BoxProjection(np.array([1.0, 1.0]), np.array([2.0, 2.0]))
    seq = SequentialProjection([custom_proj, box_proj])

    x = seq.solve(np.array([-5.0, -5.0]), max_iter=100)
    # the intersection is the box itself
    assert np.all(x >= 1.0 - 1e-10)
    assert np.all(x <= 2.0 + 1e-10)
    assert seq.proximities[-1][0] < 1e-6
