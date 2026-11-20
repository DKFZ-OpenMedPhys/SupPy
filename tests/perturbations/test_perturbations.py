import pytest
import numpy as np
from suppy.utils import FuncWrapper
from suppy.perturbations import (
    PowerSeriesGradientPerturbation,
    AdaptiveStepGradientPerturbation,
    DummyPerturbation,
)
from suppy.perturbations._base import ObjectivePerturbation
from suppy.feasibility import SequentialAMSHyperslab
from suppy.superiorization import Superiorization


@pytest.fixture
def get_test_func():
    def func(x, a):
        return a * np.sum(x**2)

    return func


@pytest.fixture
def get_test_grad():
    def grad(x, a):
        return a * 2 * x

    return grad


@pytest.fixture
def get_test_func_args():
    return [2.0]


@pytest.fixture
def get_test_grad_args():
    return [2.0]


@pytest.fixture
def get_test_perturbation(get_test_func, get_test_grad, get_test_func_args, get_test_grad_args):
    return PowerSeriesGradientPerturbation(
        get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
    )


@pytest.fixture
def get_test_perturbation_adaptive(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    return AdaptiveStepGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        func_level=1,
        epsilon=0.5,
    )


@pytest.fixture
def get_test_perturbation_noisy(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    return AdaptiveStepGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        func_level=1,
        epsilon=0.5,
        noisy=True,
    )


def test_PowerSeriesGradientPerturbation_constructor(get_test_perturbation):
    pert = get_test_perturbation

    # test objective and gradient
    assert isinstance(pert.func, FuncWrapper)
    assert isinstance(pert.grad, FuncWrapper)
    # make sure that they produce the correct results
    assert pert.func(np.array([1, 1])) == 4.0
    assert np.all(pert.grad(np.array([1, 1])) == [4.0, 4.0])

    assert pert.n_red == 1
    assert pert._k == 0
    assert pert._l == -1
    assert pert.step_size == 0.5
    assert pert.n_restart == np.inf


def test_PowerSeriesGradientPerturbation_pre_step(get_test_perturbation):
    pert = get_test_perturbation
    _ = 1
    pert.pre_step(_)

    assert pert._l == -1

    pert._l = 10
    pert._k = 3
    pert.n_restart = 3
    pert.pre_step(_)
    assert pert._l == 1


def test_PowerSeriesGradientPerturbation_function_reduction_step(get_test_perturbation):
    pert = get_test_perturbation

    x = np.array([1, 1])
    x = pert._function_reduction_step(x)

    assert np.all(x == (1 - 1 / np.sqrt(2) * np.array([1, 1])))
    assert pert._l == 0
    assert pert._k == 0


def test_PowerSeriesGradientPerturbation_single_perturbation_step(
    get_test_perturbation,
):

    pert = get_test_perturbation

    x = np.array([1, 1])
    x = pert.perturbation_step(x)

    assert np.all(x == (1 - 1 / np.sqrt(2) * np.array([1, 1])))
    assert pert._l == 0


def test_PowerSeriesGradientPerturbation_multiple_perturbation_step(
    get_test_perturbation,
):
    pert = get_test_perturbation
    assert pert._l == -1
    pert.n_red = 2
    x = np.array([1, 1])
    x = pert.perturbation_step(x)
    assert np.all(x == (1 - 1 / np.sqrt(2) - 1 / np.sqrt(8)) * np.array([1, 1]))
    assert pert._l == 1
    assert pert._k == 1


def test_AdaptiveGradientPerturbation_constructor(get_test_perturbation_adaptive):
    pert = get_test_perturbation_adaptive

    # test objective and gradient
    assert isinstance(pert.func, FuncWrapper)
    assert isinstance(pert.grad, FuncWrapper)
    # make sure that they produce the correct results
    assert pert.func(np.array([1, 1])) == 4.0
    assert np.all(pert.grad(np.array([1, 1])) == [4.0, 4.0])

    assert pert.n_red == 1
    assert pert._k == 0
    assert pert.func_level == 1.0
    assert pert.epsilon == 0.5


def test_AdaptiveGradientPerturbation_function_reduction_step(get_test_perturbation_adaptive):
    pert = get_test_perturbation_adaptive
    x = np.array([1, 1])
    x = pert._function_reduction_step(x)

    assert np.all(x == (1 - 3 / 8 * np.array([1, 1])))
    assert pert._k == 0


def test_AdaptiveGradientPerturbation_single_perturbation_step(
    get_test_perturbation_adaptive,
):
    pert = get_test_perturbation_adaptive

    x = np.array([1, 1])
    x = pert.perturbation_step(x)

    assert np.all(x == (1 - 3 / 8 * np.array([1, 1])))
    assert pert.func_level == 1.0
    assert pert.epsilon == 0.5


def test_AdaptiveGradientPerturbation_post_step(
    get_test_perturbation_adaptive,
):
    pert = get_test_perturbation_adaptive

    last_proximity_function_reduction = [0.2]
    last_proximity_basic = [0.5]

    pert.post_step(
        1,
        last_proximity_function_reduction=last_proximity_function_reduction,
        last_proximity_basic=last_proximity_basic,
    )

    assert pert.func_level == 1.0 + 0.5
    assert pert.epsilon == 0.5

    pert.func_level = 1.0  # reset

    last_proximity_basic = [0.5, 0.5]
    last_proximity_function_reduction = [1.3, 0.2]

    pert.post_step(
        1,
        last_proximity_function_reduction=last_proximity_function_reduction,
        last_proximity_basic=last_proximity_basic,
    )
    assert pert.func_level == 1.0 + (np.sqrt(1.3) - np.sqrt(0.5)) / np.sqrt(0.5)
    assert pert.epsilon == 0.5


def test_AdaptiveGradientPerturbation_post_step_noisy(
    get_test_perturbation_noisy,
):
    pert = get_test_perturbation_noisy

    last_proximity_function_reduction = [0.16]
    last_proximity_basic = [1.0]

    pert.post_step(
        1,
        last_proximity_function_reduction=last_proximity_function_reduction,
        last_proximity_basic=last_proximity_basic,
    )

    assert pert.func_level == 1.0 + 0.6
    assert pert.epsilon == 0.5

    pert.func_level = 1.0  # reset

    last_proximity_basic = [0.5, 0.5]
    last_proximity_function_reduction = [1.3, 0.2]

    pert.post_step(
        1,
        last_proximity_function_reduction=last_proximity_function_reduction,
        last_proximity_basic=last_proximity_basic,
    )
    assert pert.func_level == 1.0 + 0.5
    assert pert.epsilon == 0.5


# ---------------------------------------------------------------------------
# PowerSeriesGradientPerturbation: options and branches
# ---------------------------------------------------------------------------


def test_PowerSeriesGradientPerturbation_backtracking_loop(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    # step_size_modifier=4: the l=0 candidate is 1 - 4/sqrt(2) = -1.83 per
    # component with func 13.4 > 4 -> rejected; the l=1 candidate is
    # 1 - sqrt(2) with func 4*(3 - 2*sqrt(2)) = 0.69 <= 4 -> accepted
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        step_size_modifier=4.0,
    )
    x = np.array([1.0, 1.0])
    x_new = pert.perturbation_step(x)

    assert pert._l == 1
    assert np.allclose(x_new, (1 - np.sqrt(2)) * np.ones(2))
    assert pert.func(x_new) <= pert.func(x)


def test_PowerSeriesGradientPerturbation_n_restart_public_api(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    # bypass_objective_decrease makes _l advance exactly once per step:
    # steps 1-3 -> _l = 0, 1, 2; pre_step at _k=3 restarts the power series
    # to _l = 3 // 3 = 1, so step 4 uses step size 0.5**2
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        n_restart=3,
        bypass_objective_decrease=True,
    )
    x = np.array([1.0, 1.0])
    for _ in range(3):
        pert.pre_step(x)
        x = pert.perturbation_step(x)
    assert pert._l == 2
    assert pert._n_restarts == 0

    pert.pre_step(x)
    assert pert._l == 1  # series restarted
    assert pert._n_restarts == 1

    x_new = pert.perturbation_step(x)
    # the gradient is normalized -> the step length is exactly 0.5**2
    assert np.isclose(np.linalg.norm(x_new - x), 0.5**2)


def test_PowerSeriesGradientPerturbation_max_restarts_limit(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        n_restart=3,
        max_restarts=1,
        bypass_objective_decrease=True,
    )
    x = np.array([1.0, 1.0])
    # steps 1-3: _l = 0, 1, 2; restart at _k=3 (_l: 2 -> 1); steps 4-6
    # advance _l to 4
    for _ in range(6):
        pert.pre_step(x)
        x = pert.perturbation_step(x)
    assert pert._n_restarts == 1
    assert pert._l == 4

    # _k=6 is again a multiple of n_restart, but max_restarts is exhausted
    pert.pre_step(x)
    assert pert._l == 4  # no second restart
    assert pert._n_restarts == 1


def test_PowerSeriesGradientPerturbation_restart_value_fixed(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        n_restart=3,
        max_restarts=1,
        restart_value=0,
        bypass_objective_decrease=True,
    )
    x = np.array([1.0, 1.0])
    for _ in range(3):
        pert.pre_step(x)
        x = pert.perturbation_step(x)
    assert pert._l == 2

    pert.pre_step(x)
    assert pert._l == 0  # reset to the fixed value instead of _k // n_restart
    assert pert._n_restarts == 1


def test_PowerSeriesGradientPerturbation_restart_value_callable(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        n_restart=3,
        restart_value=lambda k: k,
        bypass_objective_decrease=True,
    )
    x = np.array([1.0, 1.0])
    for _ in range(3):
        pert.pre_step(x)
        x = pert.perturbation_step(x)

    pert.pre_step(x)
    assert pert._l == 3  # reset via the callable, evaluated at _k
    assert pert._n_restarts == 1


def test_PowerSeriesGradientPerturbation_step_size_modifier(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    # modifier 0.5 scales the first accepted candidate:
    # x - 0.5 * 0.5**0 * grad/||grad|| = 1 - 0.5/sqrt(2) per component
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        step_size_modifier=0.5,
    )
    x_new = pert.perturbation_step(np.array([1.0, 1.0]))
    assert np.allclose(x_new, (1 - 0.5 / np.sqrt(2)) * np.ones(2))
    assert pert._l == 0


def test_PowerSeriesGradientPerturbation_disable_gradient_scaling(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    # raw gradient [4, 4]: the l=0 candidate [-3, -3] increases func
    # (36 > 4) -> rejected; the l=1 candidate [1,1] - 0.5*[4,4] = [-1,-1]
    # has func 4 <= 4 -> accepted
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        disable_gradient_scaling=True,
    )
    x_new = pert.perturbation_step(np.array([1.0, 1.0]))
    assert np.allclose(x_new, [-1.0, -1.0])
    assert pert._l == 1


def test_PowerSeriesGradientPerturbation_iterative_scaling(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    # the exponent is _k (=1 in the first step) and there is no objective
    # decrease check: modifier 10 gives 1 - 10*0.5/sqrt(2) = -2.54 per
    # component although func increases (25.7 > 4)
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        iterative_scaling=True,
        step_size_modifier=10.0,
    )
    x = np.array([1.0, 1.0])
    x_new = pert.perturbation_step(x)
    assert np.allclose(x_new, (1 - 10 * 0.5 / np.sqrt(2)) * np.ones(2))
    assert pert.func(x_new) > pert.func(x)  # step was taken anyway
    assert pert._l == -1  # the power series counter is not used


def test_PowerSeriesGradientPerturbation_bypass_objective_decrease(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    # modifier 10: the candidate 1 - 10/sqrt(2) = -6.07 per component
    # increases func but is accepted without backtracking
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        step_size_modifier=10.0,
        bypass_objective_decrease=True,
    )
    x = np.array([1.0, 1.0])
    x_new = pert.perturbation_step(x)
    assert pert._l == 0  # exactly one increment, no backtracking
    assert np.allclose(x_new, (1 - 10 / np.sqrt(2)) * np.ones(2))
    assert pert.func(x_new) > pert.func(x)


def test_PowerSeriesGradientPerturbation_zero_gradient(get_test_perturbation):
    pert = get_test_perturbation
    x = np.array([0.0, 0.0])  # minimum of func -> gradient is zero
    x_new = pert.perturbation_step(x)
    assert np.array_equal(x_new, x)
    assert pert._l == -1  # no step size reduction happened


def test_PowerSeriesGradientPerturbation_tiny_step_early_exit():
    calls = {"func": 0, "grad": 0}

    def func(x):
        calls["func"] += 1
        return 2 * np.sum(x**2)

    def grad(x):
        calls["grad"] += 1
        return 4 * x

    pert = PowerSeriesGradientPerturbation(func, grad)
    # simulate a long-run state in which the power series has vanished
    pert._l = 200  # 0.5**200 = 6e-61 < 1e-14
    x = np.array([1.0, 1.0])
    x_new = pert._function_reduction_step(x)

    assert np.array_equal(x_new, x)
    # func and grad were never evaluated
    assert calls["func"] == 0
    assert calls["grad"] == 0


def test_PowerSeriesGradientPerturbation_reset(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    pert = PowerSeriesGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        n_restart=3,
        bypass_objective_decrease=True,
    )
    x = np.array([1.0, 1.0])
    for _ in range(4):
        pert.pre_step(x)
        x = pert.perturbation_step(x)
    assert (pert._k, pert._l, pert._n_restarts) == (4, 2, 1)

    pert.reset()
    assert pert._k == 0
    assert pert._l == -1
    assert pert._n_restarts == 0

    # the next step reproduces the very first step of a fresh instance
    x_new = pert.perturbation_step(np.array([1.0, 1.0]))
    assert np.allclose(x_new, (1 - 1 / np.sqrt(2)) * np.ones(2))


# ---------------------------------------------------------------------------
# AdaptiveStepGradientPerturbation: options and branches
# ---------------------------------------------------------------------------


def test_AdaptiveStepGradientPerturbation_skip_below_func_level(
    get_test_perturbation_adaptive,
):
    pert = get_test_perturbation_adaptive
    x = np.array([0.5, 0.0])  # func = 2*0.25 = 0.5 <= func_level = 1
    x_new = pert.perturbation_step(x)
    assert np.array_equal(x_new, x)


def test_AdaptiveStepGradientPerturbation_zero_gradient_skip():
    # func is above func_level at x = 0 but the gradient vanishes there
    def func(x):
        return 2 * np.sum(x**2) + 5

    def grad(x):
        return 4 * x

    pert = AdaptiveStepGradientPerturbation(func, grad, func_level=1, epsilon=0.5)
    x = np.array([0.0, 0.0])
    x_new = pert.perturbation_step(x)
    assert np.array_equal(x_new, x)


def test_AdaptiveStepGradientPerturbation_post_step_proximity_guard(
    get_test_perturbation_adaptive,
):
    pert = get_test_perturbation_adaptive

    # basic proximity below 1e-10 -> desirability 0 -> increase by epsilon
    pert.post_step(1, last_proximity_function_reduction=[0.5], last_proximity_basic=[0.0])
    assert pert.func_level == pytest.approx(1.5)

    # function reduction proximity below 1e-10 -> same
    pert.post_step(1, last_proximity_function_reduction=[0.0], last_proximity_basic=[0.5])
    assert pert.func_level == pytest.approx(2.0)


def test_AdaptiveStepGradientPerturbation_func_levels_history(
    get_test_perturbation_adaptive,
):
    pert = get_test_perturbation_adaptive
    assert pert.func_levels == [1]

    # fr < basic -> negative desirability -> increase by epsilon = 0.5 each time
    pert.post_step(1, last_proximity_function_reduction=[0.2], last_proximity_basic=[0.5])
    pert.post_step(1, last_proximity_function_reduction=[0.2], last_proximity_basic=[0.5])
    assert pert.func_levels == pytest.approx([1.0, 1.5, 2.0])


def test_AdaptiveStepGradientPerturbation_max_func_level(
    get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    pert = AdaptiveStepGradientPerturbation(
        get_test_func,
        get_test_grad,
        get_test_func_args,
        get_test_grad_args,
        func_level=1,
        epsilon=0.5,
        max_func_level=1.8,
    )
    kwargs = {"last_proximity_function_reduction": [0.2], "last_proximity_basic": [0.5]}

    pert.post_step(1, **kwargs)
    assert pert.func_level == pytest.approx(1.5)

    # 1.5 + 0.5 = 2.0 is capped at max_func_level
    pert.post_step(1, **kwargs)
    assert pert.func_level == pytest.approx(1.8)

    # once the cap is reached the update step is skipped entirely
    pert.post_step(1, **kwargs)
    assert pert.func_level == pytest.approx(1.8)
    assert pert.func_levels == pytest.approx([1.0, 1.5, 1.8])


def test_AdaptiveStepGradientPerturbation_reset(get_test_perturbation_adaptive):
    pert = get_test_perturbation_adaptive
    pert.perturbation_step(np.array([1.0, 1.0]))
    pert.post_step(1, last_proximity_function_reduction=[0.2], last_proximity_basic=[0.5])
    assert pert._k == 1
    assert pert.func_level != 1.0

    pert.reset()
    assert pert._k == 0
    assert pert.func_level == 1.0
    assert pert.func_levels == [1.0]


def test_AdaptiveStepGradientPerturbation_two_reduction_steps(
    get_test_perturbation_adaptive,
):
    pert = get_test_perturbation_adaptive
    pert.n_red = 2
    x = np.array([1.0, 1.0])
    x_new = pert.perturbation_step(x)

    # step 1: x - (4 - 1)/32 * [4, 4] = [5/8, 5/8]
    # step 2: func = 25/16, grad = [5/2, 5/2]:
    #         5/8 - (25/16 - 1)/(25/2) * 5/2 = 0.5125 per component
    assert np.allclose(x_new, 0.5125 * np.ones(2))
    assert pert._k == 1


# ---------------------------------------------------------------------------
# Base class and DummyPerturbation
# ---------------------------------------------------------------------------


def test_DummyPerturbation_identity():
    pert = DummyPerturbation()
    x = np.array([1.0, 2.0])
    x_new = pert.perturbation_step(x)
    assert np.array_equal(x_new, x)
    assert pert._k == 1
    assert pert.func(x) == 0


class _CountingPerturbation(ObjectivePerturbation):
    """Minimal concrete perturbation counting its reduction steps."""

    def __init__(self, n_red):
        super().__init__(lambda x: 0.0, [], n_red=n_red)
        self.n_calls = 0

    def _function_reduction_step(self, x):
        self.n_calls += 1
        return x


def test_ObjectivePerturbation_n_red_loop():
    pert = _CountingPerturbation(n_red=3)
    x = np.array([1.0, 1.0])
    pert.perturbation_step(x)
    assert pert.n_calls == 3
    assert pert._k == 1

    pert.perturbation_step(x)
    assert pert.n_calls == 6
    assert pert._k == 2


def test_ObjectivePerturbation_default_hooks_are_noops():
    pert = DummyPerturbation()
    x = np.array([1.0, 1.0])
    # the superiorization driver passes arbitrary keyword arguments
    pert.pre_step(x, last_proximity_basic=[0.5], last_function_value=1.0)
    pert.post_step(x, last_proximity_basic=[0.5], last_function_value=1.0)
    assert pert._k == 0  # state unchanged


# ---------------------------------------------------------------------------
# Property / invariant tests
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "make_pert",
    [
        lambda f, g, fa, ga: PowerSeriesGradientPerturbation(f, g, fa, ga),
        lambda f, g, fa, ga: PowerSeriesGradientPerturbation(f, g, fa, ga, n_red=3),
        lambda f, g, fa, ga: AdaptiveStepGradientPerturbation(
            f, g, fa, ga, func_level=1, epsilon=0.5
        ),
    ],
    ids=["power_series", "power_series_n_red_3", "adaptive"],
)
def test_perturbation_step_never_increases_objective(
    make_pert, get_test_func, get_test_grad, get_test_func_args, get_test_grad_args
):
    rng = np.random.default_rng(42)
    for _ in range(5):
        pert = make_pert(get_test_func, get_test_grad, get_test_func_args, get_test_grad_args)
        x = 3 * rng.standard_normal(4)
        x_new = pert.perturbation_step(x)
        assert pert.func(x_new) <= pert.func(x)


def test_AdaptiveStepGradientPerturbation_func_level_monotonicity(
    get_test_perturbation_adaptive,
):
    pert = get_test_perturbation_adaptive
    # (function reduction, basic) proximity pairs with improving and
    # worsening feasibility
    proximity_pairs = [(0.5, 0.5), (1.3, 0.5), (0.2, 0.5), (0.0, 0.5), (2.0, 0.1)]
    for fr, basic in proximity_pairs:
        pert.post_step(1, last_proximity_function_reduction=[fr], last_proximity_basic=[basic])
    diffs = np.diff(pert.func_levels)
    assert np.all(diffs >= pert.epsilon)


# ---------------------------------------------------------------------------
# Backend / dtype
# ---------------------------------------------------------------------------


def test_perturbation_step_int_input(get_test_perturbation):
    pert = get_test_perturbation
    # direct perturbation calls bypass the @ensure_float_array decorator on
    # solve entry points but still return float results
    x_new = pert.perturbation_step(np.array([1, 1]))
    assert np.issubdtype(x_new.dtype, np.floating)
    assert np.allclose(x_new, (1 - 1 / np.sqrt(2)) * np.ones(2))


def test_perturbation_step_cupy_backend():
    cp = pytest.importorskip("cupy")
    try:
        cp.zeros(1)  # cupy can be installed without a usable GPU
    except Exception:
        pytest.skip("cupy installed but no usable GPU")

    # backend-agnostic quadratic (avoids np.* calls on cupy arrays)
    def func(x, a):
        return a * (x @ x)

    def grad(x, a):
        return 2 * a * x

    x_np = np.array([1.5, -2.0])
    x_cp = cp.asarray(x_np)

    for make_pert in (
        lambda: PowerSeriesGradientPerturbation(func, grad, [2.0], [2.0]),
        lambda: AdaptiveStepGradientPerturbation(
            func, grad, [2.0], [2.0], func_level=1, epsilon=0.5
        ),
    ):
        res_np = make_pert().perturbation_step(x_np)
        res_cp = make_pert().perturbation_step(x_cp)
        assert isinstance(res_cp, cp.ndarray)  # no silent device -> host transfer
        assert np.allclose(cp.asnumpy(res_cp), res_np)


# ---------------------------------------------------------------------------
# Driver integration
# ---------------------------------------------------------------------------


def test_AdaptiveStepGradientPerturbation_superiorization_integration():
    # small feasibility problem mirroring the standard superiorization tests
    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    basic = SequentialAMSHyperslab(A, lb, ub)

    pert = AdaptiveStepGradientPerturbation(
        lambda x: x @ x, lambda x: 2 * x, func_level=0.5, epsilon=0.1
    )
    sup = Superiorization(basic, pert)
    x = sup.solve(np.array([2.0, 2.0]), max_iter=5)

    assert x.shape == (2,)
    assert np.all(np.isfinite(x))
    # post_step was called by the driver each iteration and raised the level
    assert len(pert.func_levels) > 1
    assert pert.func_levels[-1] >= 0.5


# ---------------------------------------------------------------------------
# Reuse of the objective value passed to pre_step by the driver
# ---------------------------------------------------------------------------


def _counting_quadratic():
    calls = []

    def func(x):
        calls.append(1)
        return np.sum(x**2)

    def grad(x):
        return 2 * x

    return func, grad, calls


def _make_schemes(func, grad):
    from suppy.perturbations._variants import BarzilaiBorweinPerturbation

    return [
        PowerSeriesGradientPerturbation(func, grad, step_size=0.5),
        AdaptiveStepGradientPerturbation(func, grad, func_level=0.1),
        BarzilaiBorweinPerturbation(func, grad, step_size=0.5),
    ]


@pytest.mark.parametrize("idx", [0, 1, 2])
def test_pre_step_known_value_saves_one_evaluation(idx):
    x = np.array([1.0, -2.0, 3.0])

    func, grad, calls = _counting_quadratic()
    pert = _make_schemes(func, grad)[idx]
    pert.pre_step(x.copy())  # no value passed: objective is evaluated
    x_ref = pert.perturbation_step(x.copy())
    n_ref = len(calls)

    func, grad, calls = _counting_quadratic()
    pert = _make_schemes(func, grad)[idx]
    pert.pre_step(x.copy(), last_function_value_basic=np.sum(x**2))
    x_new = pert.perturbation_step(x.copy())

    assert np.array_equal(x_ref, x_new)
    assert len(calls) == n_ref - 1


def test_known_value_only_used_for_first_reduction_step():
    x = np.array([1.0, -2.0, 3.0])
    func, grad, calls = _counting_quadratic()
    pert = PowerSeriesGradientPerturbation(func, grad, n_red=3, step_size=0.5)
    pert.pre_step(x.copy())
    x_ref = pert.perturbation_step(x.copy())

    pert = PowerSeriesGradientPerturbation(func, grad, n_red=3, step_size=0.5)
    pert.pre_step(x.copy(), last_function_value_basic=np.sum(x**2))
    x_new = pert.perturbation_step(x.copy())

    assert np.array_equal(x_ref, x_new)
    assert pert._known_func_value is None


def test_known_value_cleared_by_reset_and_ignored_for_none():
    func, grad, _ = _counting_quadratic()
    pert = PowerSeriesGradientPerturbation(func, grad)
    pert.pre_step(np.array([1.0]), last_function_value_basic=1.0)
    assert pert._known_func_value == 1.0
    pert.reset()
    assert pert._known_func_value is None

    pert.pre_step(None, last_function_value_basic=1.0)
    assert pert._known_func_value is None


def test_superiorization_identical_with_value_reuse():
    """Reusing the driver's objective value must not change the iterates."""

    class NoReusePSGP(PowerSeriesGradientPerturbation):
        def pre_step(self, x, *args, **kwargs):
            super().pre_step(x, *args, **kwargs)
            self._known_func_value = None  # force evaluation, as before

    A = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 2.0]])
    lb, ub = np.array([1.0, -1.0, 0.5]), np.array([2.0, 1.0, 3.0])
    results = []
    for cls in (NoReusePSGP, PowerSeriesGradientPerturbation):
        func, grad, calls = _counting_quadratic()
        sup = Superiorization(SequentialAMSHyperslab(A, lb, ub), cls(func, grad, step_size=0.8))
        x = sup.solve(np.array([3.0, 3.0]), max_iter=15, prox_tol=0, del_prox_tol=-1)
        results.append((x, sup.all_function_values, len(calls), sup._k))

    (x_a, f_a, n_a, k_a), (x_b, f_b, n_b, k_b) = results
    assert k_a == k_b
    assert np.array_equal(x_a, x_b)
    assert np.array_equal(f_a, f_b)
    assert n_b == n_a - k_b  # one evaluation fewer per iteration


def test_superiorization_fcount_equals_real_evaluations():
    """Without the cache, fcount counts actual objective evaluations."""
    A = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 2.0]])
    lb, ub = np.array([1.0, -1.0, 0.5]), np.array([2.0, 1.0, 3.0])
    func, grad, calls = _counting_quadratic()
    pert = PowerSeriesGradientPerturbation(func, grad, step_size=0.8)
    sup = Superiorization(SequentialAMSHyperslab(A, lb, ub), pert)
    sup.solve(np.array([3.0, 3.0]), max_iter=15, prox_tol=0, del_prox_tol=-1)
    assert pert.func.fcount == len(calls)


def _old_behaviour(cls):
    """
    Comparison cache in the wrapper instead of passing known values on (the
    driver is the same, so this only checks the perturbation side).
    """

    class Old(cls):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            self.func.cache = True

        def pre_step(self, x, *args, **kwargs):
            super().pre_step(x, *args, **kwargs)
            self._known_func_value = None

        def _remember_func_value(self, value):
            pass

    return Old


@pytest.mark.parametrize("idx", [0, 1, 2])
@pytest.mark.parametrize("n_red", [1, 3])
def test_superiorization_store_function_reduction_no_extra_evaluations(idx, n_red):
    """Passing values on needs no more evaluations than the cache, same
    results.
    """
    from suppy.perturbations._variants import BarzilaiBorweinPerturbation

    classes = [
        PowerSeriesGradientPerturbation,
        AdaptiveStepGradientPerturbation,
        BarzilaiBorweinPerturbation,
    ]
    kwargs = [{"step_size": 0.8}, {"func_level": 0.1}, {"step_size": 0.8}][idx]
    A = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 2.0]])
    lb, ub = np.array([1.0, -1.0, 0.5]), np.array([2.0, 1.0, 3.0])

    results = []
    for cls in (_old_behaviour(classes[idx]), classes[idx]):
        real = []

        def func(x):
            real.append(1)
            return np.sum(x**2)

        pert = (
            cls(func, lambda x: 2 * x, n_red=n_red, **kwargs)
            if idx != 1
            else cls(func, lambda x: 2 * x, **kwargs)
        )
        sup = Superiorization(SequentialAMSHyperslab(A, lb, ub), pert)
        x = sup.solve(
            np.array([3.0, 3.0]),
            max_iter=15,
            prox_tol=0,
            del_prox_tol=-1,
            store_function_reduction=True,
        )
        results.append((x, sup.all_function_values, len(real)))

    (x_old, f_old, n_old), (x_new, f_new, n_new) = results
    assert np.array_equal(x_old, x_new)
    assert np.array_equal(f_old, f_new)
    assert n_new == n_old


def test_func_value_after_step_returns_stored_value_once():
    func, grad, calls = _counting_quadratic()
    pert = PowerSeriesGradientPerturbation(func, grad, step_size=0.5)
    x = np.array([1.0, -2.0, 3.0])
    pert.pre_step(x, last_function_value_basic=np.sum(x**2))
    x_new = pert.perturbation_step(x)
    n = len(calls)
    assert pert.func_value_after_step(x_new) == np.sum(x_new**2)
    assert len(calls) == n  # value from the line search, no evaluation
    assert pert.func_value_after_step(x_new) == np.sum(x_new**2)
    assert len(calls) == n + 1  # second call evaluates


def test_superiorization_reuses_value_at_fixed_point():
    """
    If the iteration returns to its starting point (perturbation moves x,
    the
    basic step projects it back), the objective is not evaluated again.
    """
    A = np.array([[1.0, 1.0], [1.0, -1.0], [0.5, 2.0]])
    lb, ub = np.array([1.0, -1.0, 0.5]), np.array([2.0, 1.0, 3.0])
    func, grad, calls = _counting_quadratic()
    pert = AdaptiveStepGradientPerturbation(func, grad, func_level=0.1)
    sup = Superiorization(SequentialAMSHyperslab(A, lb, ub), pert)
    x = sup.solve(np.array([3.0, 3.0]), max_iter=40, prox_tol=0, del_prox_tol=-1)
    # x reaches the fixed point [0.5, 0.5] after 3 iterations
    assert np.allclose(x, [0.5, 0.5])
    assert sup._k == 40
    assert len(calls) == 4
