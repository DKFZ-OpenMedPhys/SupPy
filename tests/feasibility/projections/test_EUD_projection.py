import numpy as np
import pytest
from suppy.projections import EUDProjection


def test_EUDProjection_constructor():

    eud = EUDProjection(a=5, EUD_max=20, relaxation=1.5)
    assert eud.relaxation == 1.5
    assert eud.a == 5
    assert eud.level == 20


def test_EUDProjection_func():

    eud = EUDProjection(a=5, EUD_max=20, relaxation=1.5)
    x = np.array([1, 2, 3])
    assert np.allclose(eud.func_call(x), (276 / 3) ** (1 / 5))


def test_EUDProjection_grad():

    eud = EUDProjection(a=5, EUD_max=20, relaxation=1.5)
    x = np.array([1, 2, 3])
    assert np.allclose(eud.grad_call(x), 276 ** (-4 / 5) * x ** (4) / 3 ** (1 / 5))


def test_EUDProjection_level_diff():
    eud = EUDProjection(a=5, EUD_max=20, relaxation=1.5)
    x = np.array([1, 2, 3])
    assert eud.level_diff(x) == (276 / 3) ** (1 / 5) - 20


def test_EUDProjection_proximity():
    proximity_measures = [("p_norm", 2), "max_norm"]
    eud = EUDProjection(a=5, EUD_max=20, relaxation=1.5)
    x = np.array([1, 2, 3])
    assert eud._proximity(x, proximity_measures)[0] == 0
    assert eud._proximity(x, proximity_measures)[1] == 0
    x = np.array([25, 25, 25])
    assert np.abs(eud._proximity(x, proximity_measures)[0] - 25) < 1e-10
    assert np.abs(eud._proximity(x, proximity_measures)[1] - 5) < 1e-10


# ---------------------------------------------------------------------------
# EUDProjection: options and branches
# ---------------------------------------------------------------------------


def test_EUDProjection_public_proximity_and_flag():
    proximity_measures = [("p_norm", 2), "max_norm"]
    eud = EUDProjection(a=5, EUD_max=20)
    # the public wrapper returns an array
    prox = eud.proximity(np.array([25.0, 25.0, 25.0]), proximity_measures)
    assert isinstance(prox, np.ndarray)
    assert np.abs(prox[0] - 25) < 1e-10

    eud_off = EUDProjection(a=5, EUD_max=20, proximity_flag=False)
    prox_off = eud_off.proximity(np.array([25.0, 25.0, 25.0]), proximity_measures)
    assert np.array_equal(prox_off, np.zeros(2))


def test_EUDProjection_project_inside_level_is_noop():
    eud = EUDProjection(a=5, EUD_max=20)
    x = np.array([1.0, 2.0, 3.0])  # EUD ~ 2.46 < 20
    assert np.allclose(eud.project(x.copy()), x)


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_EUDProjection_repeated_projects_converge_to_level():
    # a single subgradient step overshoots slightly for non-uniform points;
    # repeated projections converge onto the EUD_max level set
    eud = EUDProjection(a=5, EUD_max=20)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = 20 + 10 * rng.random(3)  # EUD well above 20
        f_before = eud.func_call(x)
        x = eud.project(x)
        assert eud.func_call(x) < f_before  # each step reduces the EUD
        for _ in range(100):
            x = eud.project(x)
        assert np.abs(eud.func_call(x) - 20) < 1e-6
