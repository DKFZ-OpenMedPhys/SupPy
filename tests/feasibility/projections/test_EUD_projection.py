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
    eud = EUDProjection(a=5, EUD_max=20, relaxation=1.5)
    x = np.array([1, 2, 3])
    assert eud._proximity(x) == 0
    x = np.array([25, 25, 25])
    assert abs(eud._proximity(x) - 25) < 1e-10
