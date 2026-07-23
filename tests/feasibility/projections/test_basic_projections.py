import numpy as np
import pytest
from suppy.projections import (
    BoxProjection,
    WeightedBoxProjection,
    HalfspaceProjection,
    BandProjection,
    BallProjection,
    MaxDVHProjection,
    MinDVHProjection,
)


def test_BoxProjection_datatype_error():
    """Test to check that box projections does not work with int datatype."""

    # check type error

    lb = np.array([0.1, 0.1])
    ub = np.array([1.2, 1.2])

    Box_Proj_error = BoxProjection(lb, ub)  # without relaxation

    x = np.array([0, 2])
    # with pytest.raises(np._core_exceptions._UFuncOutputCastingError):
    proj = Box_Proj_error.project(x)
    assert np.all(proj == np.array([0, 1]))  # mathematically wrong, result is because of int array


def test_BoxProjection_no_relaxation():
    """Test to check that box projections work without relaxation."""

    x = np.array([0, 1, 2])

    # run it on a 3D geometry
    lb = np.array([0, 0, 0])
    ub = np.array([1, 1, 1])
    Box_Proj = BoxProjection(lb, ub)  # without relaxation

    assert np.all(Box_Proj.lb == lb)
    assert np.all(Box_Proj.ub == ub)
    assert Box_Proj.relaxation == 1.0
    assert Box_Proj.idx == np.s_[:]

    proj = Box_Proj.project(x)

    # project onto relaxed 3D box
    x = np.array([-1, 0.5, 2])
    proj = Box_Proj.project(x)

    # check that projection has correct data type and is modified in place
    assert isinstance(proj, np.ndarray)
    assert np.all(proj == x)
    assert np.all(proj == np.array([0, 0.5, 1]))


def test_BoxProjection_relaxation():
    """Test to check that box projections work with relaxation."""
    lb = np.array([0, 0, 0])
    ub = np.array([1, 1, 1])

    Box_Proj2 = BoxProjection(lb, ub, relaxation=1.5)  # with relaxation

    assert np.all(Box_Proj2.lb == lb)
    assert np.all(Box_Proj2.ub == ub)
    assert Box_Proj2.relaxation == 1.5
    assert Box_Proj2.idx == np.s_[:]

    # with relaxation
    x = np.array([-1, 0.5, 2])
    proj = Box_Proj2.project(x)
    assert np.all(proj == np.array([0.5, 0.5, 0.5]))

    # visualizing the box should raise an error
    with pytest.raises(ValueError):
        Box_Proj2.visualize()

    with pytest.raises(ValueError):
        Box_Proj2.get_xy()


def test_Box_projection_idx():
    """
    Test to check that box projections work with indexing for lower
    dimensions.
    """

    lb = np.array([0, 0])
    ub = np.array([1, 1])

    # run on a 2D example (but x is 3D)
    Box_Proj3 = BoxProjection(lb, ub, idx=[0, 1], relaxation=1.5)

    assert np.all(Box_Proj3.lb == lb)
    assert np.all(Box_Proj3.ub == ub)
    assert Box_Proj3.relaxation == 1.5
    assert Box_Proj3.idx == [0, 1]

    x = np.array([-1, 0.5, 2])
    proj = Box_Proj3.project(x)

    # check that projection has correct data type and is modified in place
    assert np.all(proj == np.array([0.5, 0.5, 2]))
    assert isinstance(proj, np.ndarray)

    # check that projection does not map correct point when already in set
    assert np.all(Box_Proj3.project(proj) == np.array([0.5, 0.5, 2]))

    # visualizing the box works this time
    Box_Proj3.visualize()
    # get xy also can be called properly
    Box_Proj3.get_xy()


def test_BoxProjection_proximity():
    lb = np.array([0, 0])
    ub = np.array([1, 1])
    box_proj = BoxProjection(lb, ub, idx=[0, 1], relaxation=1.5)
    x = np.array([-1, 0.5, 2])

    prox_measures = []
    no_prox = box_proj.proximity(x, prox_measures)
    assert no_prox.size == 0

    prox_measures = [("p_norm", 2), "max_norm"]

    prox = box_proj.proximity(x, prox_measures)

    assert np.abs(prox[0] - 0.5) < 1e-10
    assert np.abs(prox[1] - 1) < 1e-10


def test_HalfspaceProjection_datatype_error():
    """
    Test to check that integer arrays do not work properly on halfspace
    projections but float arrays do and produce expected result.
    """
    a = np.array([1, 1, 1])
    b = 2
    x = np.array([1, 1, 1])
    Halfspace_Proj = HalfspaceProjection(a, b)
    # with pytest.raises(_UFuncOutputCastingError):
    with pytest.raises(np._core._exceptions._UFuncOutputCastingError):
        proj = Halfspace_Proj.project(x)


def test_HalfspaceProjection_no_relaxation():
    """Test the halfspace projection without relaxation."""
    a = np.array([1, 1, 1])
    b = 2
    x = np.array([1, 1, 1])
    Halfspace_Proj = HalfspaceProjection(a, b)

    assert np.all(Halfspace_Proj.a == a)
    assert Halfspace_Proj.b == b
    assert Halfspace_Proj.relaxation == 1.0
    assert Halfspace_Proj.idx == np.s_[:]

    x = np.array([1, 1, 1], dtype="float64")
    proj = Halfspace_Proj.project(x)

    # check that projection has correct data type and is modified in place
    assert np.all(proj == x)
    assert isinstance(proj, np.ndarray)
    assert np.all((proj - np.array([2 / 3, 2 / 3, 2 / 3])) < 1e-10)


def test_HalfspaceProjection_relaxation():
    """Test the halfspace projection with relaxation."""

    a = np.array([1, 1, 1])
    b = 2
    x = np.array([1, 1, 1], dtype="float64")
    Halfspace_Proj2 = HalfspaceProjection(a, b, relaxation=1.5)

    assert np.all(Halfspace_Proj2.a == a)
    assert Halfspace_Proj2.b == b
    assert Halfspace_Proj2.relaxation == 1.5
    assert Halfspace_Proj2.idx == np.s_[:]

    proj = Halfspace_Proj2.project(x)
    assert np.all(proj == 1 / 2 * np.array([1, 1, 1]))

    # visualizing the halfspace should raise an error
    with pytest.raises(ValueError):
        Halfspace_Proj2.visualize()

    with pytest.raises(ValueError):
        Halfspace_Proj2.get_xy()


def test_HalfspaceProjection_idx():
    """Test the halfspace projection with indexing for lower dimensions."""

    # run on a 2D example (but x is 3D)
    a = np.array([1, 1])
    b = 1
    x = np.array([1, 1, 1], dtype="float64")

    Halfspace_Proj3 = HalfspaceProjection(a, b, idx=[0, 1], relaxation=1.5)

    assert np.all(Halfspace_Proj3.a == a)
    assert Halfspace_Proj3.b == b
    assert Halfspace_Proj3.relaxation == 1.5
    assert Halfspace_Proj3.idx == [0, 1]

    proj = Halfspace_Proj3.project(x)

    # check that projection has correct data type and is modified in place
    assert np.all(proj == np.array([1 / 4, 1 / 4, 1]))
    Halfspace_Proj3.visualize()
    Halfspace_Proj3.get_xy()


def test_BandProjection_datatype_error():
    """
    Test the band projection to check that integer arrays do not work
    properly.
    """
    a = np.array([1.2, 1.2, 1.2])
    lb = -2
    ub = 2
    x = np.array([1, 1, 1])
    Band_Proj = BandProjection(a, lb, ub)
    with pytest.raises(np._core._exceptions._UFuncOutputCastingError):
        proj = Band_Proj.project(x)


def test_BandProjection_no_relaxation():
    """Test the band projection without relaxation on float arrays."""
    a = np.array([1, 1, 1])
    lb = -2
    ub = 2
    Band_Proj = BandProjection(a, lb, ub)

    x = np.array([1, 1, 1], dtype="float64")
    proj = Band_Proj.project(x)

    # check that projection has correct data type and is modified in place
    assert np.all(proj == x)
    assert isinstance(proj, np.ndarray)
    assert np.all((proj - 2 / 3 * np.array([1, 1, 1])) < 1e-10)

    x = -1 * np.array([1, 1, 1], dtype="float64")
    proj = Band_Proj.project(x)
    assert np.all((proj + 2 / 3 * np.array([1, 1, 1])) < 1e-10)


def test_BandProjection_relaxation():
    """Test the band projection with relaxation on float arrays."""
    # with relaxation 1.5
    a = np.array([1, 1, 1])
    lb = -2
    ub = 2

    x = np.array([1, 1, 1], dtype="float64")
    Band_Proj2 = BandProjection(a, lb, ub, relaxation=1.5)
    proj = Band_Proj2.project(x)
    assert np.all(proj == 1 / 2 * np.array([1, 1, 1]))

    # visualizing the band should raise an error
    with pytest.raises(ValueError):
        Band_Proj2.visualize()

    with pytest.raises(ValueError):
        Band_Proj2.get_xy()


def test_BandProjection_idx():
    """Test the band projection with indexing for lower dimensions."""

    # run on a 2D example (but x is 3D)
    a = np.array([1, 1])
    lb = -1
    ub = 1
    x = np.array([1, 1, 1], dtype="float64")
    Band_Proj3 = BandProjection(a, lb, ub, idx=[0, 1], relaxation=1.5)
    proj = Band_Proj3.project(x)
    assert np.all((proj - np.array([1 / 4, 1 / 4, 1]) < 1e-10))
    Band_Proj3.visualize()
    Band_Proj3.get_xy()


def test_BallProjection_datatype_error():
    """
    Test the ball projection to check that integer arrays do not work
    properly.
    """

    center = np.array([1, 1])
    radius = 1
    x = np.array([0, 0])
    Ball_Proj = BallProjection(center, radius)

    with pytest.raises(np._core._exceptions._UFuncOutputCastingError):
        proj = Ball_Proj.project(x)


def test_BallProjection_no_relaxation():
    """Test the ball projection without relaxation on float arrays."""
    center = np.array([1, 1])
    radius = 1
    Ball_Proj = BallProjection(center, radius)

    assert np.all(Ball_Proj.center == center)
    assert Ball_Proj.radius == radius
    assert Ball_Proj.relaxation == 1.0
    assert Ball_Proj.idx == np.s_[:]

    x = np.array([0, 0], dtype="float64")
    proj = Ball_Proj.project(x)

    # check that projection has correct data type and is modified in place
    assert np.all(proj == x)
    assert isinstance(proj, np.ndarray)
    assert np.all(proj == (1 - 1 / np.sqrt(2)) * np.array([1, 1]))


def test_BallProjection_relaxation():
    """Test the ball projection with relaxation on float arrays."""
    center = np.array([1, 1])
    radius = 1
    # with relaxation 1.5
    x = np.array([0, 0], dtype="float64")
    Ball_Proj2 = BallProjection(center, radius, relaxation=1.5)

    assert np.all(Ball_Proj2.center == center)
    assert Ball_Proj2.radius == radius
    assert Ball_Proj2.relaxation == 1.5
    assert Ball_Proj2.idx == np.s_[:]

    proj = Ball_Proj2.project(x)
    assert np.all(proj == 3 / 2 * (1 - 1 / np.sqrt(2)) * np.array([1, 1]))

    # visualizing the ball should not raise an error
    Ball_Proj2.visualize()
    Ball_Proj2.get_xy()


def test_BallProjection_idx():
    """Test the ball projection with indexing for lower dimensions."""

    # run on a 3D example (but x is 2D)
    center = np.array([1, 1])
    radius = 1
    x = np.array([0, 0, 0], dtype="float64")
    Ball_Proj3 = BallProjection(center, radius, idx=[0, 1], relaxation=1.5)

    assert np.all(Ball_Proj3.center == center)
    assert Ball_Proj3.radius == radius
    assert Ball_Proj3.relaxation == 1.5
    assert Ball_Proj3.idx == [0, 1]

    proj = Ball_Proj3.project(x)
    assert np.all(proj == 3 / 2 * (1 - 1 / np.sqrt(2)) * np.array([1, 1, 0]))


def test_MaxDVHProjection_constructor():
    """Test the constructor of the DVH projection."""
    # check that the constructor works properly

    d_max = 5
    max_overflow = 10
    DVH_Proj = MaxDVHProjection(d_max, max_overflow)
    assert DVH_Proj.d_max == d_max
    assert DVH_Proj.max_percentage == max_overflow
    assert DVH_Proj.idx == np.s_[:]
    with pytest.raises(ValueError):
        MaxDVHProjection(d_max, max_overflow, idx=np.array([True, False]))

    DVH_Proj2 = MaxDVHProjection(d_max, max_overflow, idx=np.array([0, 1, 2]))
    assert DVH_Proj2.d_max == d_max
    assert DVH_Proj2.max_percentage == max_overflow
    assert np.array_equal(DVH_Proj2.idx, np.array([0, 1, 2]))


def test_MaxDVHProjection_project():
    """Test the projection of the DVH projection."""
    d_max = 5
    max_overflow = 0.1  # 10% can overflow
    DVH_Proj = MaxDVHProjection(d_max, max_overflow)
    x = np.arange(10)
    proj = DVH_Proj.project(x)
    assert np.all(proj == np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 9]))


def test_MaxDVHProjection_project_idxs():
    """Test the projection of the DVH projection on subset."""
    d_max = 5
    max_overflow = 0.1  # 10% can overflow
    idx = np.arange(0, 10, 1)
    DVH_Proj = MaxDVHProjection(d_max, max_overflow, idx=idx)
    x = np.arange(11)
    proj = DVH_Proj.project(x)
    assert np.all(proj == np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 9, 10]))


def test_MaxDVHProjection_project_odd_overflow():
    """
    Test the projection for the DVH projection with an overflow percentage
    that does not divide the number of elemetns.
    """
    d_max = 5
    max_overflow = 0.05  # at most 5% of the volume are allowed to receive a dose higher than d_max
    DVH_Proj = MaxDVHProjection(d_max, max_overflow)
    x = np.arange(10)
    proj = DVH_Proj.project(x)
    assert np.all(proj == np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5]))


def test_MaxDVHProjection_project_odd_overflow_idxs():
    """
    Test the projection for the DVH projection with an overflow percentage
    that does not divide the number of elemetns.
    """
    d_max = 5
    max_overflow = 0.05  # at most 5% of the volume are allowed to receive a dose higher than d_max
    idx = np.arange(0, 10, 1)
    DVH_Proj = MaxDVHProjection(d_max, max_overflow, idx=idx)
    x = np.arange(11)
    proj = DVH_Proj.project(x)
    assert np.all(proj == np.array([0, 1, 2, 3, 4, 5, 5, 5, 5, 5, 10]))


def test_MaxDVHProjection_proximity():
    d_max = 10
    max_overflow = 0.05
    DVH_Proj = MaxDVHProjection(d_max, max_overflow)
    x = np.ones(99) * 11
    proximity_measures = [("p_norm", 2), "max_norm"]
    prox = DVH_Proj.proximity(x, proximity_measures)
    # test p norm
    assert np.abs(prox[0] - 95 / 99) < 1e-10
    assert np.abs(prox[1] - 1) < 1e-10

    x_2 = np.ones(99) * 9
    prox_2 = DVH_Proj.proximity(x_2, proximity_measures)
    assert np.abs(prox_2[0] - 0) < 1e-10
    assert np.abs(prox_2[1] - 0) < 1e-10

    x[:10] = 9

    prox_3 = DVH_Proj.proximity(x, proximity_measures)
    assert np.abs(prox_3[0] - 85 / 99) < 1e-10
    assert np.abs(prox_3[1] - 1) < 1e-10


def test_MaxDVHProjection_proximity_idxs():
    d_max = 10
    max_overflow = 0.05
    DVH_Proj = MaxDVHProjection(d_max, max_overflow, idx=np.arange(0, 99))
    x = np.ones(100) * 11
    proximity_measures = [("p_norm", 2), "max_norm"]
    prox = DVH_Proj.proximity(x, proximity_measures)
    # test p norm
    assert np.abs(prox[0] - 95 / 99) < 1e-10
    assert np.abs(prox[1] - 1) < 1e-10

    x_2 = np.ones(100) * 9
    prox_2 = DVH_Proj.proximity(x_2, proximity_measures)
    assert np.abs(prox_2[0] - 0) < 1e-10
    assert np.abs(prox_2[1] - 0) < 1e-10

    x[:10] = 9

    prox_3 = DVH_Proj.proximity(x, proximity_measures)
    assert np.abs(prox_3[0] - 85 / 99) < 1e-10
    assert np.abs(prox_3[1] - 1) < 1e-10


def test_MinDVHProjection_constructor():
    """Test the constructor of the DVH projection."""
    # check that the constructor works properly

    d_min = 5
    min_overflow = 10
    DVH_Proj = MinDVHProjection(d_min, min_overflow)
    assert DVH_Proj.d_min == d_min
    assert DVH_Proj.min_percentage == min_overflow
    assert DVH_Proj.idx == np.s_[:]

    DVH_Proj2 = MinDVHProjection(d_min, min_overflow, idx=np.array([0, 1, 2]))
    assert DVH_Proj2.d_min == d_min
    assert DVH_Proj2.min_percentage == min_overflow
    assert np.array_equal(DVH_Proj2.idx, [0, 1, 2])


def test_MinDVHProjection_project():
    """Test the projection of the DVH projection."""
    d_min = 5
    min_percentage = 0.9  # 50% can overflow
    DVH_Proj = MinDVHProjection(d_min, min_percentage)
    x = np.arange(10)
    proj = DVH_Proj.project(x)
    assert np.all(proj == np.array([0, 5, 5, 5, 5, 5, 6, 7, 8, 9]))


def test_MinDVHProjection_project_idxs():
    """Test the projection of the DVH projection on subset."""
    d_min = 5
    min_percentage = 0.9  # 50% can overflow
    idx = np.arange(0, 10, 1)
    DVH_Proj = MinDVHProjection(d_min, min_percentage, idx=idx)
    x = np.arange(11)
    proj = DVH_Proj.project(x)
    assert np.all(proj == np.array([0, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10]))


def test_MinDVHProjection_project_odd_overflow():
    """
    Test the projection for the DVH projection with an overflow percentage
    that does not divide the number of elemetns.
    """
    d_min = 5
    min_overflow = 0.95  # 5% can overflow, but since we have only 10 elements for x this means no element can overflow
    DVH_Proj = MinDVHProjection(d_min, min_overflow)
    x = np.arange(10)
    proj = DVH_Proj.project(x)
    assert np.all(proj == np.array([5, 5, 5, 5, 5, 5, 6, 7, 8, 9]))


def test_MinDVHProjection_project_odd_overflow_idxs():
    """
    Test the projection for the DVH projection with an overflow percentage
    that does not divide the number of elemetns.
    """
    d_min = 5
    min_overflow = 0.95  # 5% can overflow, but since we have only 10 elements for x this means no element can overflow
    idx = np.arange(0, 10, 1)
    DVH_Proj = MinDVHProjection(d_min, min_overflow, idx=idx)
    x = np.arange(11)
    proj = DVH_Proj.project(x)
    assert np.all(proj == np.array([5, 5, 5, 5, 5, 5, 6, 7, 8, 9, 10]))


def test_MinDVHProjection_proximity():
    d_min = 5
    min_percentage = 0.95
    DVH_Proj = MinDVHProjection(d_min, min_percentage)
    x = np.ones(101) * 4
    proximity_measures = [("p_norm", 2), "max_norm"]
    prox = DVH_Proj.proximity(x, proximity_measures)
    # test p norm
    assert np.abs(prox[0] - 96 / 101) < 1e-10
    assert np.abs(prox[1] - 1) < 1e-10

    x_2 = np.ones(101) * 6
    prox_2 = DVH_Proj.proximity(x_2, proximity_measures)
    assert np.abs(prox_2[0] - 0) < 1e-10
    assert np.abs(prox_2[1] - 0) < 1e-10

    x[:10] = 6

    prox_3 = DVH_Proj.proximity(x, proximity_measures)
    assert np.abs(prox_3[0] - 86 / 101) < 1e-10
    assert np.abs(prox_3[1] - 1) < 1e-10


def test_MinDVHProjection_proximity_idxs():
    d_min = 5
    min_percentage = 0.95
    DVH_Proj = MinDVHProjection(d_min, min_percentage, idx=np.arange(0, 101))
    x = np.ones(102) * 4
    proximity_measures = [("p_norm", 2), "max_norm"]
    prox = DVH_Proj.proximity(x, proximity_measures)
    # test p norm
    assert np.abs(prox[0] - 96 / 101) < 1e-10
    assert np.abs(prox[1] - 1) < 1e-10

    x_2 = np.ones(102) * 6
    prox_2 = DVH_Proj.proximity(x_2, proximity_measures)
    assert np.abs(prox_2[0] - 0) < 1e-10
    assert np.abs(prox_2[1] - 0) < 1e-10

    x[:10] = 6

    prox_3 = DVH_Proj.proximity(x, proximity_measures)
    assert np.abs(prox_3[0] - 86 / 101) < 1e-10
    assert np.abs(prox_3[1] - 1) < 1e-10


# ---------------------------------------------------------------------------
# WeightedBoxProjection: options and branches
# ---------------------------------------------------------------------------


def test_WeightedBoxProjection_constructor_normalizes_weights():
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    proj = WeightedBoxProjection(lb, ub, weights=np.array([3.0, 1.0]))
    assert np.allclose(proj.weights, np.array([0.75, 0.25]))
    assert proj.relaxation == 1
    assert proj.idx == np.s_[:]


def test_WeightedBoxProjection_project():
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    proj = WeightedBoxProjection(lb, ub, weights=np.array([3.0, 1.0]))
    # x = [2, 2]: full clip target is [1, 1]; the weighted step moves only a
    # weight-fraction toward it: [2 - 0.75, 2 - 0.25] = [1.25, 1.75]
    x = proj.project(np.array([2.0, 2.0]))
    assert np.allclose(x, np.array([1.25, 1.75]))


def test_WeightedBoxProjection_full_project():
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    proj = WeightedBoxProjection(lb, ub, weights=np.array([3.0, 1.0]))
    # _full_project clips all the way regardless of the weights
    x = proj._full_project(np.array([2.0, 2.0]))
    assert np.allclose(x, np.array([1.0, 1.0]))


def test_WeightedBoxProjection_proximity():
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    proj = WeightedBoxProjection(lb, ub, weights=np.array([3.0, 1.0]))
    # the residual is measured against the weighted (partial) step, not the
    # full clip: res = [0.75, 0.25]
    # p_norm 2 -> 0.75*0.75**2 + 0.25*0.25**2 = 0.4375; max_norm -> 0.75
    prox = proj.proximity(np.array([2.0, 2.0]), [("p_norm", 2), "max_norm"])
    assert np.abs(prox[0] - 0.4375) < 1e-10
    assert np.abs(prox[1] - 0.75) < 1e-10


def test_WeightedBoxProjection_idx():
    lb = np.array([0.0, 0.0])
    ub = np.array([1.0, 1.0])
    proj = WeightedBoxProjection(lb, ub, weights=np.array([1.0, 1.0]), idx=[0, 1])
    # only the first two entries are projected (half-way, equal weights 0.5)
    x = proj.project(np.array([2.0, 2.0, 5.0]))
    assert np.allclose(x, np.array([1.5, 1.5, 5.0]))


# ---------------------------------------------------------------------------
# Shared branches: proximity flag and invalid measures
# ---------------------------------------------------------------------------


def test_BoxProjection_proximity_flag_false():
    proj = BoxProjection(np.array([0.0, 0.0]), np.array([1.0, 1.0]), proximity_flag=False)
    prox = proj.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


def test_BoxProjection_invalid_proximity_measure():
    proj = BoxProjection(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    with pytest.raises(ValueError):
        proj.proximity(np.array([2.0, 2.0]), ["not_a_measure"])


def test_MinDVHProjection_bool_idx_error():
    # boolean indexing is rejected (mirror of the MaxDVH constructor test)
    with pytest.raises(ValueError):
        MinDVHProjection(5, 0.9, idx=np.array([True, False]))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_proj, x_feasible",
    [
        (
            lambda: BoxProjection(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
            np.array([0.5, 0.5]),
        ),
        (
            lambda: HalfspaceProjection(np.array([1.0, 1.0, 1.0]), 2),
            np.array([0.5, 0.5, 0.5]),
        ),
        (lambda: BandProjection(np.array([1.0, 1.0]), -2, 2), np.array([0.5, 0.5])),
        (lambda: BallProjection(np.array([1.0, 1.0]), 1), np.array([0.5, 0.5])),
    ],
    ids=["box", "halfspace", "band", "ball"],
)
def test_BasicProjection_project_idempotent_on_feasible(make_proj, x_feasible):
    proj = make_proj()
    assert np.allclose(proj.project(x_feasible.copy()), x_feasible)


@pytest.mark.parametrize(
    "make_proj",
    [
        lambda: BoxProjection(np.array([0.0, 0.0]), np.array([1.0, 1.0])),
        lambda: BandProjection(np.array([1.0, 1.0]), -1, 1),
        lambda: BallProjection(np.array([1.0, 1.0]), 1),
    ],
    ids=["box", "band", "ball"],
)
def test_BasicProjection_projected_point_is_feasible(make_proj):
    # after projection the proximity must vanish (the point is in the set)
    proj = make_proj()
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        x_p = proj.project(x)
        prox = proj.proximity(x_p, [("p_norm", 2)])
        assert prox[0] < 1e-20


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_BasicProjection_cupy_backend():
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    cases = [
        (
            lambda xp: BoxProjection(xp.asarray([0.0, 0.0]), xp.asarray([1.0, 1.0])),
            np.array([2.0, -1.0]),
        ),
        (
            lambda xp: HalfspaceProjection(xp.asarray([1.0, 1.0]), 1),
            np.array([2.0, 2.0]),
        ),
        (lambda xp: BallProjection(xp.asarray([1.0, 1.0]), 1), np.array([3.0, 3.0])),
    ]
    for make_proj, x_np in cases:
        res_np = make_proj(np).project(x_np.copy())
        res_cp = make_proj(cp).project(cp.asarray(x_np))
        assert isinstance(res_cp, cp.ndarray)  # no silent device -> host transfer
        assert np.allclose(cp.asnumpy(res_cp), res_np)
