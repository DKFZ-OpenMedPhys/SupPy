import numpy as np
import pytest
from suppy.projections import (
    BallProjection,
    SequentialProjection,
    SimultaneousProjection,
    BlockIterativeProjection,
)


@pytest.fixture
def get_SequentialProjection_input():
    center_1 = np.array([0, 0])
    center_2 = np.array([1, 1])
    radius = 1

    x0 = np.array([2, 2], dtype=np.float64)

    return x0, SequentialProjection(
        [BallProjection(center_1, radius), BallProjection(center_2, radius)]
    )


def test_SequentialProjection_proximity(get_SequentialProjection_input):
    """Test the proximity function of the SequentialProjection class."""
    proximity_measures = [("p_norm", 2), "max_norm"]
    x0, seq_proj = get_SequentialProjection_input

    assert seq_proj.proximity(x0, proximity_measures)[0] == 6 * (
        1 - 1 / np.sqrt(2)
    )  # check distance
    assert seq_proj.proximity(x0, proximity_measures)[1] == (2 * np.sqrt(2) - 1)  # check distance

    assert np.array_equal(
        x0, np.array([2, 2], dtype=np.float64)
    )  # make sure that proximity call did not change x0


def test_SequentialProjection_project(get_SequentialProjection_input):
    """Test the project function of the SequentialProjection class."""
    x0, seq_proj = get_SequentialProjection_input
    x = seq_proj.project(x0)
    assert np.array_equal(x, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))  # check projection


def test_SequentialProjection_solve(get_SequentialProjection_input):
    """Test the solve function of the SequentialProjection class."""
    x0, seq_proj = get_SequentialProjection_input

    assert np.array_equal(
        np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), seq_proj.solve(x0.copy())
    )  # check that x0 is not changed


@pytest.fixture
def get_SequentialProjection_input2():
    center_1 = np.array([0, 0])
    center_2 = np.array([1, 1])
    radius = 1

    x0 = np.array([2, 2], dtype=np.float64)

    return x0, SequentialProjection(
        [BallProjection(center_2, radius), BallProjection(center_1, radius)],
        control_seq=[1, 0],
    )


def test_SequentialProjection_control_seq(get_SequentialProjection_input):
    """Test the control_seq attribute of the SequentialProjection class."""
    x0, seq_proj = get_SequentialProjection_input
    x = seq_proj.project(x0.copy())
    assert np.array_equal(x, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))  # check projection


@pytest.fixture
def get_SimultaneousProjection_input():
    center_1 = np.array([0, 0])
    center_2 = np.array([0, 1])
    radius = 1

    x0 = np.array([0, 3], dtype=np.float64)

    return x0, SimultaneousProjection(
        [BallProjection(center_1, radius), BallProjection(center_2, radius)]
    )


def test_SimultaneousProjection_proximity(get_SimultaneousProjection_input):
    """Test the proximity function of the SimultaneousProjection class."""
    x0, sim_proj = get_SimultaneousProjection_input
    proximity_measures = [("p_norm", 2), "max_norm"]
    assert sim_proj.proximity(x0, proximity_measures)[0] == 2.5  # check distance
    assert sim_proj.proximity(x0, proximity_measures)[1] == 2  # check distance
    assert np.array_equal(
        x0, np.array([0, 3], dtype=np.float64)
    )  # make sure that proximity call did not change x0


def test_SimultaneousProjection_project(get_SimultaneousProjection_input):
    """Test the project function of the SimultaneousProjection class."""
    x0, sim_proj = get_SimultaneousProjection_input
    x = sim_proj.project(x0)
    assert np.array_equal(x, np.array([0, 1.5]))  # check projection


@pytest.fixture
def get_sequential_BlockIterativeProjection_input():
    center_1 = np.array([0, 0])
    center_2 = np.array([1, 1])
    radius = 1

    x0 = np.array([2, 2], dtype=np.float64)

    return x0, BlockIterativeProjection(
        [BallProjection(center_1, radius), BallProjection(center_2, radius)],
        weights=np.eye(2),
    )


@pytest.fixture
def get_simultaneous_BlockIterativeProjection_input():
    center_1 = np.array([0, 0])
    center_2 = np.array([0, 1])
    radius = 1

    x0 = np.array([0, 3], dtype=np.float64)

    return x0, BlockIterativeProjection(
        [BallProjection(center_1, radius), BallProjection(center_2, radius)],
        weights=np.array([[1 / 2, 1 / 2]]),
    )


def test_sequential_BlockIterativeProjection_proximity(
    get_sequential_BlockIterativeProjection_input,
):
    """Test the proximity function of the BlockIterativeProjection class."""
    x0, BlockIterProj = get_sequential_BlockIterativeProjection_input
    proximity_measures = [("p_norm", 2), "max_norm"]
    assert BlockIterProj.proximity(x0, proximity_measures)[0] == 6 * (
        1 - 1 / np.sqrt(2)
    )  # check distance
    assert BlockIterProj.proximity(x0, proximity_measures)[1] == (
        2 * np.sqrt(2) - 1
    )  # check distance
    assert np.array_equal(
        x0, np.array([2, 2], dtype=np.float64)
    )  # make sure that proximity call did not change x0


def test_sequential_BlockIterativeProjection_project(
    get_sequential_BlockIterativeProjection_input,
):
    """Test the project function of the BlockIterativeProjection class."""
    x0, BlockIterProj = get_sequential_BlockIterativeProjection_input
    x = BlockIterProj.project(x0)
    assert np.array_equal(x, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]))  # check projection


def test_simultaneous_BlockIterativeProjection_proximity(
    get_simultaneous_BlockIterativeProjection_input,
):
    """Test the proximity function of the BlockIterativeProjection class."""
    x0, BlockIterProj = get_simultaneous_BlockIterativeProjection_input
    proximity_measures = [("p_norm", 2), "max_norm"]
    assert BlockIterProj.proximity(x0, proximity_measures)[0] == 2.5  # check distance
    assert BlockIterProj.proximity(x0, proximity_measures)[1] == 2  # check distance
    assert np.array_equal(
        x0, np.array([0, 3], dtype=np.float64)
    )  # make sure that proximity call did not change x0


def test_simultaneous_BlockIterativeProjection_project(
    get_simultaneous_BlockIterativeProjection_input,
):
    """Test the project function of the BlockIterativeProjection class."""
    x0, BlockIterProj = get_simultaneous_BlockIterativeProjection_input
    x = BlockIterProj.project(x0)
    assert np.array_equal(x, np.array([0, 1.5]))  # check projection
