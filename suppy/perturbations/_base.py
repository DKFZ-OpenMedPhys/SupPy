"""Base class for perturbations applied to feasibility seeking algorithms."""
from abc import ABC, abstractmethod
from typing import Callable, List
import numpy as np
import numpy.typing as npt
from suppy.utils import FuncWrapper

try:
    import cupy as cp

    NO_GPU = False
except ImportError:
    NO_GPU = True
    cp = np


class Perturbation(ABC):
    """
    Abstract base class for perturbations applied to feasibility seeking
    algorithms.
    """

    @abstractmethod
    def perturbation_step(self, x: npt.NDArray) -> np.ndarray:
        """
        Perform a perturbation step.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be perturbed.

        Returns
        -------
        npt.NDArray
            The perturbed array.
        """


class ObjectivePerturbation(Perturbation, ABC):
    """
    Base class for perturbations performed by decreasing an objective
    function.

    Parameters
    ----------
    func : Callable
        The objective function to be perturbed.
    func_args : List
        The arguments to be passed to the objective function.
    n_red : int, optional
        The number of reduction steps to perform in one perturbation iteration (default is 1).

    Attributes
    ----------
    func : FuncWrapper
        A wrapped version of the objective function with its arguments.
    n_red : int
        The number of reduction steps to perform.
    _k : int
        Keeps track of the number of performed perturbations.
    """

    def __init__(self, func: Callable, func_args: List, n_red: int = 1):
        self.func = FuncWrapper(func, func_args)
        self.n_red = n_red
        self._k = 0

    def perturbation_step(self, x: npt.NDArray) -> np.ndarray:
        """
        Perform n_red perturbation steps on the input array.

        Parameters
        ----------
        x : npt.NDArray
            The input array to be perturbed.

        Returns
        -------
        npt.NDArray
            The perturbed array after applying the reduction steps.
        """
        self._k += 1
        n = 0
        while n < self.n_red:
            x = self._function_reduction_step(x)
            n += 1
        return x

    @abstractmethod
    def _function_reduction_step(self, x: npt.NDArray) -> np.ndarray:
        """
        Abstract method to perform that should implement the individual
        function reduction steps on the input array.
        Needs to be implemented by subclasses.

        Parameters
        ----------
        x : npt.NDArray
            Input array on which the reduction step is to be performed.

        Returns
        -------
        npt.NDArray
            The array after the reduction step has been applied.
        """

    def pre_step(self, x: npt.NDArray, *args, **kwargs) -> None:
        """
        If required perform any form of step previous to each
        perturbation in each iteration.

        This method is intended to be overridden by subclasses to implement
        specific pre-step logic. By default, it does nothing.

        Parameters
        ----------
        x : npt.NDArray
            Current iterate.
        """

    def post_step(self, x: npt.NDArray, *args, **kwargs) -> None:
        """
        If required perform any form of step after each perturbation in each
        iteration.

        This method is intended to be overridden by subclasses to implement
        specific post-step logic. By default, it does nothing.

        Parameters
        ----------
        x : npt.NDArray
            Current iterate.
        """

    def reset(self) -> None:
        """Reset the perturbation to its initial state."""
        self._k = 0


class DummyPerturbation(ObjectivePerturbation):
    """Dummy perturbation that does not change the input."""

    def __init__(self):
        def _dummy_func(x):
            """Always returns 0."""
            return 0

        func_args = []

        self.func = FuncWrapper(_dummy_func, func_args)
        self.n_red = 1
        self._k = 0  # keeps track of the number of performed perturbations

    def _function_reduction_step(self, x: npt.NDArray) -> np.ndarray:
        return x


class GradientPerturbation(ObjectivePerturbation, ABC):
    """
    A class for perturbations performed by decreasing an objective function
    using the gradient.

    Parameters
    ----------
    func : Callable
        The objective function to be perturbed.
    grad : Callable
        The gradient function of the objective function.
    func_args : List
        The arguments to be passed to the objective function.
    grad_args : List
        The arguments to be passed to the gradient function.
    n_red : int, optional
        The reduction factor, by default 1.

    Attributes
    ----------
    func : FuncWrapper
        A wrapped version of the objective function with its arguments.
    grad : FuncWrapper
        A wrapped gradient function with its arguments.
    n_red : int
        The number of reduction steps to perform.
    _k : int
        Keeps track of the number of performed perturbations.
    """

    def __init__(
        self,
        func: Callable,
        grad: Callable,
        func_args: List,
        grad_args: List,
        n_red: int = 1,
    ):
        super().__init__(func, func_args, n_red)
        self.grad = FuncWrapper(grad, grad_args)


class AdaptiveStepGradientPerturbation(GradientPerturbation):
    """
    Objective function perturbation using gradient descent with adaptive
    step size reduction.

    Parameters
    ----------
    func : Callable
        The function to be reduced.
    grad : Callable
        The gradient of the function to be reduced.
    func_args : List, optional
        Additional arguments to be passed to the function, by default [].
    grad_args : List, optional
        Additional arguments to be passed to the gradient function, by default [].
    func_level : float, optional
        Value above which to perform a gradient step using a subgradient projection like step.
    max_func_level : float, optional
        Upper bound for the func_level; once reached the update step is skipped, by default np.inf.
    epsilon : float, optional
        Default value to update func_level by.
    noisy : bool, optional
        Changes the behavior of the update step.

    References
    ----------
    - [1] https://doi.org/10.1007/s11075-021-01229-z
    """

    def __init__(
        self,
        func: Callable,
        grad: Callable,
        func_args: List | None = None,
        grad_args: List | None = None,
        func_level: float = 0.5,
        max_func_level: float = np.inf,
        epsilon: float = 1e-6,
        noisy: bool = False,
    ):
        if func_args is None:
            func_args = []
        if grad_args is None:
            grad_args = []
        super().__init__(func, grad, func_args, grad_args, n_red=1)
        self.func_level = func_level
        self.max_func_level = max_func_level
        self.epsilon = epsilon
        self.update_direction = -1 if noisy else 1
        self.func_levels = [func_level]

    def _function_reduction_step(self, x: npt.NDArray) -> np.ndarray:
        """
        Perform a function reduction step using gradient descent with a step
        size based on a subgradient projection.

        Parameters
        ----------
        x : npt.NDArray
            The current point in the algorithm.

        Returns
        -------
        npt.NDArray
            The updated point after performing the reduction step.
        """
        grad_eval = self.grad(x)
        func_eval = self.func(x)
        if grad_eval @ grad_eval <= 0 or func_eval <= self.func_level:
            # if gradient is zero (or negative) or the function value is below alpha, skip
            return x
        return x - (func_eval - self.func_level) / (grad_eval @ grad_eval) * grad_eval

    def post_step(self, x: npt.NDArray, *args, **kwargs) -> None:
        """Update func_level after each step."""
        last_proximity_function_reduction = kwargs.get("last_proximity_function_reduction")[0]
        last_proximity_basic = kwargs.get("last_proximity_basic")[0]

        if last_proximity_basic < 1e-10 or last_proximity_function_reduction < 1e-10:
            desirability = 0
        else:
            desirability = (
                np.sqrt(last_proximity_function_reduction) - np.sqrt(last_proximity_basic)
            ) / np.sqrt(last_proximity_basic)

        self.func_level = self.func_level + max(
            self.epsilon, self.update_direction * self.func_level * desirability
        )
        self.func_levels.append(self.func_level)

    def reset(self) -> None:
        """Reset the perturbation to its initial state."""
        super().reset()
        self._l = -1


class PowerSeriesGradientPerturbation(GradientPerturbation):
    """
    Objective function perturbation using gradient descent with step size
    reduction according to a power series.
    Has the option to restart the power series after a certain number of
    steps.

    Parameters
    ----------
    func : Callable
        The function to be reduced.
    grad : Callable
        The gradient of the function to be reduced.
    func_args : List, optional
        Additional arguments to be passed to the function, by default [].
    grad_args : List, optional
        Additional arguments to be passed to the gradient function, by default [].
    n_red : int, optional
        The number of reductions, by default 1.
    step_size : float, optional
        The step size for the gradient descent, by default 0.5.
    step_size_modifier : float, optional
        Scaling factor for the step size power series, by default 1.0.
    n_restart : int, optional
        The number of steps after which to restart the power series, by default -1 (no restart).
    disable_gradient_scaling : bool, optional
        If true, skip the normalization of the gradient, by default False.
    iterative_scaling : bool, optional
        If true, the power series is scaled by the iteration k without checking whether this actually decreases the objective, by default False.
    bypass_objective_decrease : bool, optional
        If true, the check whether the objective function value actually decreases is bypassed, by default False.
    """

    def __init__(
        self,
        func: Callable,
        grad: Callable,
        func_args: List | None = None,
        grad_args: List | None = None,
        n_red: int = 1,
        step_size: float = 0.5,
        step_size_modifier: float = 1.0,
        n_restart: int = -1,
        disable_gradient_scaling: bool = False,
        iterative_scaling: bool = False,
        bypass_objective_decrease: bool = False,
    ):
        if func_args is None:
            func_args = []
        if grad_args is None:
            grad_args = []
        super().__init__(func, grad, func_args, grad_args, n_red)
        self.step_size = step_size
        self.step_size_modifier = step_size_modifier
        self._l = -1
        self.n_restart = np.inf if n_restart == -1 else n_restart
        self.disable_gradient_scaling = disable_gradient_scaling
        self.iterative_scaling = iterative_scaling
        self.bypass_objective_decrease = bypass_objective_decrease
        self._last_grad_norm = 1

    def _function_reduction_step(self, x: npt.NDArray) -> np.ndarray:
        """
        Perform a function reduction step using gradient descent.

        Parameters
        ----------
        x : npt.NDArray
            The current point in the algorithm.

        Returns
        -------
        npt.NDArray
            The updated point after performing the reduction step.
        """
        # quick check if the step size is already zero, if so skip the rest of the calculations
        if self.step_size_modifier * self.step_size**self._l * self._last_grad_norm <= 1e-14:
            return x

        xp = cp if isinstance(x, cp.ndarray) else np
        grad_eval = self.grad(x)
        func_eval = self.func(x)

        if grad_eval @ grad_eval <= 0:
            return x

        grad_norm = 1 if self.disable_gradient_scaling else xp.linalg.norm(grad_eval)
        self._last_grad_norm = grad_norm

        if not self.iterative_scaling:
            while True:
                self._l += 1
                x_ln = (
                    x - self.step_size_modifier * self.step_size**self._l * grad_eval / grad_norm
                )
                if self.bypass_objective_decrease or self.func(x_ln) <= func_eval:
                    return x_ln
        else:
            return (
                x - self.step_size_modifier * self.step_size ** (self._k) * grad_eval / grad_norm
            )

    def pre_step(self, x: npt.NDArray, *args, **kwargs) -> None:
        """
        Resets the power series after n steps.

        Parameters
        ----------
        x : npt.NDArray
            Current iterate.
        """
        if self._k <= 0:
            return
        if self._k % self.n_restart == 0:
            self._l = self._k // self.n_restart

    def reset(self) -> None:
        """Reset the perturbation to its initial state."""
        super().reset()
        self._l = -1
