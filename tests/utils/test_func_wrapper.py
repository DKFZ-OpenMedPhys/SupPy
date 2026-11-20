import numpy as np
from suppy.utils import FuncWrapper


def _counting_func():
    calls = []

    def func(x, a):
        calls.append(1)
        return a * np.sum(x**2)

    return func, calls


def test_FuncWrapper_cache_default_returns_stored_value():
    func, calls = _counting_func()
    wrapped = FuncWrapper(func, [2.0])
    x = np.array([1.0, 2.0])
    assert wrapped(x) == 10.0
    assert wrapped(x.copy()) == 10.0
    assert len(calls) == 1  # second call served from the cache
    assert wrapped.fcount == 2


def test_FuncWrapper_cache_detects_in_place_change():
    func, calls = _counting_func()
    wrapped = FuncWrapper(func, [1.0])
    x = np.array([1.0, 1.0])
    assert wrapped(x) == 2.0
    x += 1  # same object, new values
    assert wrapped(x) == 8.0
    assert len(calls) == 2


def test_FuncWrapper_without_cache_always_evaluates():
    func, calls = _counting_func()
    wrapped = FuncWrapper(func, [2.0], cache=False)
    x = np.array([1.0, 2.0])
    assert wrapped(x) == 10.0
    assert wrapped(x) == 10.0
    assert len(calls) == 2
    assert wrapped.fcount == 2


def test_FuncWrapper_default_args():
    wrapped = FuncWrapper(lambda x: x.sum())
    assert wrapped.args == []
    assert wrapped(np.array([1.0, 2.0])) == 3.0
