import numpy as np
import pytest
from suppy.projections import (
    BallProjection,
    SequentialProjection,
    SimultaneousProjection,
    BlockIterativeProjection,
    StringAveragedProjection,
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


def test_SequentialProjection_control_seq(get_SequentialProjection_input2):
    """Test the control_seq attribute of the SequentialProjection class."""
    x0, seq_proj = get_SequentialProjection_input2
    # control_seq=[1, 0] visits ball(0, 0) first, then ball(1, 1):
    # [2, 2] -> [1/sqrt(2), 1/sqrt(2)] which already lies inside ball(1, 1)
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


def test_ProjectionMethod_solve(get_SequentialProjection_input):
    """Test the solve function of the ProjectionMethod class."""
    x0, seq_proj = get_SequentialProjection_input

    assert np.array_equal(
        np.array([1 / np.sqrt(2), 1 / np.sqrt(2)]), seq_proj.solve(x0.copy())
    )  # check that x0 is not changed
    assert np.array_equal(seq_proj.proximities, [[6 * (1 - 1 / np.sqrt(2))], [0]])

    assert seq_proj.all_x is None

    x0_2, seq_proj_2 = get_SequentialProjection_input

    seq_proj_2.solve(x0_2.copy(), storage=True)
    assert np.array_equal(seq_proj_2.proximities, [[6 * (1 - 1 / np.sqrt(2))], [0]])

    assert np.array_equal(seq_proj_2.all_x, [x0_2, np.array([1 / np.sqrt(2), 1 / np.sqrt(2)])])


# ---------------------------------------------------------------------------
# StringAveragedProjection: options and branches
# ---------------------------------------------------------------------------


@pytest.fixture
def get_two_balls():
    return [
        BallProjection(np.array([0.0, 0.0]), 1),
        BallProjection(np.array([1.0, 1.0]), 1),
    ]


def test_StringAveragedProjection_project(get_two_balls):
    proj = StringAveragedProjection(get_two_balls, strings=[[0], [1]])
    assert np.allclose(proj.weights, np.array([0.5, 0.5]))
    # from [2, 2]: string [0] -> [1/sqrt(2)]*2, string [1] -> [1 + 1/sqrt(2)]*2;
    # equal-weight average: [0.5 + 1/sqrt(2)]*2
    x = proj.project(np.array([2.0, 2.0]))
    assert np.allclose(x, (0.5 + 1 / np.sqrt(2)) * np.ones(2))


def test_StringAveragedProjection_custom_weights_normalized(get_two_balls):
    # weights [4, 1] are normalized to [0.8, 0.2]:
    # 0.8*[1/sqrt(2)]*2 + 0.2*[1 + 1/sqrt(2)]*2 = [0.2 + 1/sqrt(2)]*2
    proj = StringAveragedProjection(
        get_two_balls, strings=[[0], [1]], weights=np.array([4.0, 1.0])
    )
    assert np.allclose(proj.weights, np.array([0.8, 0.2]))
    x = proj.project(np.array([2.0, 2.0]))
    assert np.allclose(x, (0.2 + 1 / np.sqrt(2)) * np.ones(2))


def test_StringAveragedProjection_single_string_matches_sequential(get_two_balls):
    # one string covering all projections is exactly the sequential sweep
    proj_sa = StringAveragedProjection(get_two_balls, strings=[[0, 1]])
    proj_s = SequentialProjection(get_two_balls)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        assert np.allclose(proj_sa.project(x.copy()), proj_s.project(x.copy()))


def test_StringAveragedProjection_solve(get_two_balls):
    # the ball intersection is nonempty -> solve reaches proximity ~0
    proj = StringAveragedProjection(get_two_balls, strings=[[0], [1]])
    proj.solve(np.array([2.0, 2.0]), max_iter=200)
    assert proj.proximities[-1][0] < 1e-6


# ---------------------------------------------------------------------------
# BlockIterativeProjection: options and branches
# ---------------------------------------------------------------------------


def test_BlockIterativeProjection_weights_length_mismatch(get_two_balls):
    # each weight row must have one entry per projection
    with pytest.raises(ValueError):
        BlockIterativeProjection(get_two_balls, weights=[np.array([1.0])])


def test_BlockIterativeProjection_weights_row_sum_error(get_two_balls):
    # each weight row must sum to 1
    with pytest.raises(ValueError):
        BlockIterativeProjection(get_two_balls, weights=[np.array([0.5, 0.6])])


def test_BlockIterativeProjection_plain_list_weights():
    # plain lists are allowed per the type hint (used to crash on list.sum())
    center_1 = np.array([0.0, 0.0])
    center_2 = np.array([0.0, 1.0])
    balls = [BallProjection(center_1, 1), BallProjection(center_2, 1)]
    proj_list = BlockIterativeProjection(balls, weights=[[0.5, 0.5]])
    proj_arr = BlockIterativeProjection(balls, weights=np.array([[0.5, 0.5]]))
    x = np.array([0.0, 3.0])
    assert np.allclose(proj_list.project(x.copy()), proj_arr.project(x.copy()))


# ---------------------------------------------------------------------------
# ProjectionMethod.solve: options and branches
# ---------------------------------------------------------------------------


def test_ProjectionMethod_solve_max_iter_zero(get_SequentialProjection_input):
    x0, seq_proj = get_SequentialProjection_input
    x = seq_proj.solve(x0.copy(), max_iter=0)
    assert np.array_equal(x, x0)
    assert len(seq_proj.proximities) == 1  # only the initial proximity


def test_ProjectionMethod_solve_storage_iters_int(get_SequentialProjection_input):
    x0, seq_proj = get_SequentialProjection_input
    # the problem converges after one projection; storage_iters=2 keeps only
    # index 0 while the proximity history is recorded fully
    seq_proj.solve(x0.copy(), storage=True, storage_iters=2)
    assert seq_proj.all_x.shape == (1, 2)
    assert np.array_equal(seq_proj.all_x[0], x0)
    assert len(seq_proj.proximities) == 2


def test_ProjectionMethod_solve_alternative_stopping_criterion(get_SequentialProjection_input):
    x0, seq_proj = get_SequentialProjection_input
    # the initial call can stop the solve before any iteration
    x = seq_proj.solve(
        x0.copy(),
        max_iter=100,
        alternative_stopping_criterion_initial_call=lambda x, alg: True,
    )
    assert np.array_equal(x, x0)
    assert len(seq_proj.proximities) == 1


def test_ProjectionMethod_solve_stall_exit_on_disjoint_sets():
    # disjoint balls: the sequential projection settles into a 2-cycle, the
    # proximity plateaus above prox_tol and the del_prox criterion stops
    # the iteration early
    balls = [
        BallProjection(np.array([0.0, 0.0]), 1),
        BallProjection(np.array([4.0, 0.0]), 1),
    ]
    seq_proj = SequentialProjection(balls)
    seq_proj.solve(np.array([2.0, 2.0]), max_iter=100, del_prox_n=3)
    assert len(seq_proj.proximities) - 1 < 100  # stopped early
    assert seq_proj.proximities[-1][0] > 1e-6  # still infeasible at the stop


def test_ProjectionMethod_invalid_proximity_measure(get_SequentialProjection_input):
    x0, seq_proj = get_SequentialProjection_input
    with pytest.raises(ValueError):
        seq_proj.proximity(x0, ["not_a_measure"])


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_proj",
    [
        lambda balls: SequentialProjection(balls),
        lambda balls: SimultaneousProjection(balls),
        lambda balls: StringAveragedProjection(balls, strings=[[0], [1]]),
        lambda balls: BlockIterativeProjection(balls, weights=np.eye(2)),
    ],
    ids=["sequential", "simultaneous", "string_averaged", "block_iterative"],
)
def test_ProjectionMethod_fejer_monotone(make_proj, get_two_balls):
    proj = make_proj(get_two_balls)
    z = np.array([0.5, 0.5])  # point in the intersection of both balls
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 4 * rng.standard_normal(2)
        for _ in range(10):
            d_before = np.linalg.norm(x - z)
            x = proj.project(x)
            assert np.linalg.norm(x - z) <= d_before + 1e-12


@pytest.mark.parametrize(
    "make_proj",
    [
        lambda balls: SequentialProjection(balls),
        lambda balls: SimultaneousProjection(balls),
        lambda balls: StringAveragedProjection(balls, strings=[[0], [1]]),
        lambda balls: BlockIterativeProjection(balls, weights=np.eye(2)),
    ],
    ids=["sequential", "simultaneous", "string_averaged", "block_iterative"],
)
def test_ProjectionMethod_project_idempotent_on_feasible(make_proj, get_two_balls):
    proj = make_proj(get_two_balls)
    x = np.array([0.5, 0.5])  # inside both balls
    assert np.allclose(proj.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_ProjectionMethod_cupy_backend():
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    def make_balls(xp):
        return [
            BallProjection(xp.asarray([0.0, 0.0]), 1),
            BallProjection(xp.asarray([1.0, 1.0]), 1),
        ]

    for make_proj in (
        lambda balls: SequentialProjection(balls),
        lambda balls: SimultaneousProjection(balls),
    ):
        x_np = make_proj(make_balls(np)).project(np.array([2.0, 2.0]))
        x_cp = make_proj(make_balls(cp)).project(cp.asarray([2.0, 2.0]))
        assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
        assert np.allclose(cp.asnumpy(x_cp), x_np)


@pytest.mark.parametrize("measure", [("p_norm", 2), ("p_norm", 1), "max_norm"])
@pytest.mark.parametrize("weights", [None, np.array([0.25, 0.75])])
def test_weighted_proximity_cupy_backend(measure, weights):
    """The weighted p_norm proximity must not mix numpy weights with cupy
    proxs.
    """
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    def make_balls(xp):
        return [
            BallProjection(xp.asarray([0.0, 0.0]), 1),
            BallProjection(xp.asarray([1.0, 1.0]), 1),
        ]

    block_weights = [[0.5, 0.5]] if weights is None else [list(weights)]
    for make_proj in (
        lambda balls: SimultaneousProjection(balls, weights=weights),
        lambda balls: BlockIterativeProjection(balls, weights=block_weights),
    ):
        p_np = make_proj(make_balls(np)).proximity(np.array([5.0, 5.0]), [measure])
        p_cp = make_proj(make_balls(cp)).proximity(cp.asarray([5.0, 5.0]), [measure])
        assert isinstance(p_cp, cp.ndarray)  # no silent device -> host transfer
        assert np.allclose(cp.asnumpy(p_cp), p_np)
