import numpy as np
import pytest
from suppy.projections import SubgradientProjection


@pytest.fixture
def get_func_grad_no_args():
    def func(x):
        return np.linalg.norm(x)

    def grad(x):
        return x / np.linalg.norm(x)

    return (func, grad)


@pytest.fixture
def get_func_grad_args():
    def func(x, a, b):
        return a * np.linalg.norm(x) + b

    def grad(x, a):
        return a * x / np.linalg.norm(x)

    return (func, grad)


def test_subgradientProjection_constructor_no_args(get_func_grad_no_args):

    func, grad = get_func_grad_no_args

    subgrad = SubgradientProjection(func, grad, relaxation=1.5)
    assert subgrad.func == func
    assert subgrad.grad == grad
    assert subgrad.func_args == []
    assert subgrad.grad_args == []
    assert subgrad.level == 0
    assert subgrad.relaxation == 1.5
    assert subgrad.idx == np.s_[:]


def test_subgradientProjection_constructor_args(get_func_grad_args):

    func, grad = get_func_grad_args

    subgrad = SubgradientProjection(func, grad, func_args=[2, 1], grad_args=[2])
    assert subgrad.func == func
    assert subgrad.grad == grad
    assert subgrad.func_args == [2, 1]
    assert subgrad.grad_args == [2]
    assert subgrad.level == 0
    assert subgrad.relaxation == 1
    assert subgrad.idx == np.s_[:]


def test_subgradientProjection_constructor_idx(get_func_grad_no_args):

    func, grad = get_func_grad_no_args

    subgrad = SubgradientProjection(func, grad, idx=[0, 1])
    assert subgrad.func == func
    assert subgrad.grad == grad
    assert subgrad.func_args == []
    assert subgrad.grad_args == []
    assert subgrad.level == 0
    assert subgrad.relaxation == 1
    assert subgrad.idx == [0, 1]


def test_subgradientProjection_func_grad_call_no_args(get_func_grad_no_args):

    func, grad = get_func_grad_no_args

    subgrad = SubgradientProjection(func, grad)
    x = np.array([1.0, 2.0, 3.0])
    assert subgrad.func_call(x) == 3.7416573867739413
    assert np.all(
        abs(subgrad.grad_call(x) - np.array([0.26726124, 0.53452248, 0.80178373])) < 1e-8
    )


def test_subgradientProjection_func_grad_call_args(get_func_grad_args):

    func, grad = get_func_grad_args

    subgrad = SubgradientProjection(func, grad, func_args=[2, 1], grad_args=[2])
    x = np.array([1.0, 2.0, 3.0])
    assert subgrad.func_call(x) == 8.483314773547882
    assert np.all(
        abs(subgrad.grad_call(x) - np.array([0.53452248, 1.06904497, 1.60356745])) < 1e-8
    )


def test_subgradientProjection_project_no_args(get_func_grad_no_args):

    func, grad = get_func_grad_no_args

    subgrad = SubgradientProjection(func, grad)
    x = np.array([1.0, 2.0, 3.0])
    x_n = subgrad.project(x)
    assert np.array_equal(x_n, x)
    assert np.all(abs(x_n - np.array([0, 0, 0])) < 1e-8)


def test_subgradientProjection_project_args(get_func_grad_args):

    func, grad = get_func_grad_args

    subgrad = SubgradientProjection(func, grad, func_args=[2, 1], grad_args=[2])
    x = np.array([1.0, 2.0, 3.0])
    x_n = subgrad.project(x)
    assert np.array_equal(x_n, x)
    assert np.all(
        abs(
            x_n - (np.array([1.0, 2.0, 3.0]) * (1 - 2 * (2 * np.sqrt(14) + 1) / (4 * np.sqrt(14))))
        )
        < 1e-8
    )


def test_subgradientProjection_level_set_call(get_func_grad_no_args):

    func, grad = get_func_grad_no_args

    subgrad = SubgradientProjection(func, grad, level=3)
    x_1 = np.array([1.0, 1.0, 1.0])
    x_n = subgrad.project(x_1)
    assert np.array_equal(x_n, x_1)
    assert np.all(abs(x_n - np.array([1.0, 1.0, 1.0])) < 1e-8)

    x_2 = np.array([1.0, 2.0, 3.0])
    x_n = subgrad.project(x_2)
    assert np.array_equal(x_n, x_2)
    assert np.all(
        abs(x_n - (np.array([1.0, 2.0, 3.0]) * (3 / np.linalg.norm(np.array([1.0, 2.0, 3.0])))))
        < 1e-8
    )


def test_subgradientProjection_idx_call(get_func_grad_no_args):

    func, grad = get_func_grad_no_args

    subgrad = SubgradientProjection(func, grad, idx=[0, 1])
    x = np.array([1.0, 2.0, 3.0])
    x_n = subgrad.project(x)
    assert np.array_equal(x_n, x)
    assert np.all(abs(x_n - np.array([0, 0, 3])) < 1e-8)


def test_level_diff(get_func_grad_no_args):

    func, grad = get_func_grad_no_args

    subgrad = SubgradientProjection(func, grad, level=3)
    x = np.array([1.0, 2.0, 3.0])
    assert abs(subgrad.level_diff(x) - (np.sqrt(14) - 3)) < 1e-8
    x = np.array([1.0, 1.0, 1.0])
    assert abs(subgrad.level_diff(x) - (np.sqrt(3) - 3)) < 1e-8


def test_proximity(get_func_grad_no_args):

    func, grad = get_func_grad_no_args
    proximity_measures = [("p_norm", 2), "max_norm"]
    subgrad = SubgradientProjection(func, grad, level=3)
    x = np.array([1.0, 2.0, 3.0])
    assert np.abs(subgrad.proximity(x, proximity_measures)[0] - (np.sqrt(14) - 3) ** 2) < 1e-8
    assert np.abs(subgrad.proximity(x, proximity_measures)[1] - (np.sqrt(14) - 3)) < 1e-8

    x = np.array([1.0, 1.0, 1.0])
    assert np.abs(subgrad.proximity(x, proximity_measures)[0] - 0) < 1e-8
    assert np.abs(subgrad.proximity(x, proximity_measures)[1] - 0) < 1e-8


# ---------------------------------------------------------------------------
# SubgradientProjection: options and branches
# ---------------------------------------------------------------------------


def test_subgradientProjection_relaxed_project(get_func_grad_no_args):
    func, grad = get_func_grad_no_args
    # the unrelaxed projection (level 0) maps x to the origin; the outer
    # relaxation halves the step: x + 0.5 * (0 - x) = 0.5 * x
    subgrad = SubgradientProjection(func, grad, relaxation=0.5)
    x = np.array([1.0, 2.0, 3.0])
    x_n = subgrad.project(x)
    assert np.allclose(x_n, 0.5 * np.array([1.0, 2.0, 3.0]))


def test_subgradientProjection_zero_gradient_guard():
    # func is above the level but the gradient vanishes: the guard skips the
    # (undefined) step and returns x unchanged
    subgrad = SubgradientProjection(lambda x: 1.0, lambda x: np.zeros_like(x))
    x = np.array([1.0, 2.0, 3.0])
    x_n = subgrad.project(x)
    assert np.array_equal(x_n, np.array([1.0, 2.0, 3.0]))


def test_subgradientProjection_proximity_flag_false(get_func_grad_no_args):
    func, grad = get_func_grad_no_args
    subgrad = SubgradientProjection(func, grad, level=3, proximity_flag=False)
    prox = subgrad.proximity(np.array([100.0, 100.0]), [("p_norm", 2), "max_norm"])
    assert np.array_equal(prox, np.zeros(2))


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


def test_subgradientProjection_lands_on_level_set(get_func_grad_no_args):
    # for the norm objective the subgradient step maps any exterior point
    # exactly onto the level set
    func, grad = get_func_grad_no_args
    subgrad = SubgradientProjection(func, grad, level=3)
    rng = np.random.default_rng(42)
    for _ in range(5):
        x = rng.standard_normal(3)
        x *= 6 / np.linalg.norm(x)  # scale to norm 6 (outside the level set)
        x_n = subgrad.project(x)
        assert np.abs(np.linalg.norm(x_n) - 3) < 1e-10


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_subgradientProjection_cupy_backend():
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    # backend-agnostic norm (works on numpy and cupy arrays)
    def func(x):
        return (x @ x) ** 0.5

    def grad(x):
        return x / (x @ x) ** 0.5

    x_np = SubgradientProjection(func, grad, level=3).project(np.array([1.0, 2.0, 3.0]))

    subgrad_cp = SubgradientProjection(func, grad, level=3)
    x_cp = subgrad_cp.project(cp.asarray([1.0, 2.0, 3.0]))
    assert isinstance(x_cp, cp.ndarray)  # no silent device -> host transfer
    assert np.allclose(cp.asnumpy(x_cp), x_np)
