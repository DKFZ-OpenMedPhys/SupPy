from abc import ABC, abstractmethod
from typing import Callable, List
from suppy.utils import FuncWrapper
import numpy as np
import numpy.typing as npt

try:
    import cupy as cp

    no_gpu = False
except ImportError:
    no_gpu = True
    cp = np


class Perturbation(ABC):
    """
    Abstract base class for perturbations applied to feasibility seeking
    algorithms.
    """

    @abstractmethod
    def perturbation_step(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Perform a perturbation step.

        Parameters
        ----------
        x : npt.ArrayLike
            The input array to be perturbed.

        Returns
        -------
        npt.ArrayLike
            The perturbed array.
        """
        pass


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

    def __init__(self, func: Callable, func_args: List, n_red=1):
        self.func = FuncWrapper(func, func_args)
        self.n_red = n_red
        self._k = 0  # keeps track of the number of performed perturbations

    def perturbation_step(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Perform n_red perturbation steps on the input array.

        Parameters
        ----------
        x : npt.ArrayLike
            The input array to be perturbed.

        Returns
        -------
        npt.ArrayLike
            The perturbed array after applying the reduction steps.
        """

        self._k += 1
        n = 0
        while n < self.n_red:
            x = self._function_reduction_step(x)
            n += 1
        return x

    @abstractmethod
    def _function_reduction_step(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Abstract method to perform that should implement the individual
        function reduction steps on the input array.
        Needs to be implemented by subclasses.

        Parameters
        ----------
        x : npt.ArrayLike
            Input array on which the reduction step is to be performed.

        Returns
        -------
        npt.ArrayLike
            The array after the reduction step has been applied.
        """
        pass

    def pre_step(self):
        """
        If required perform any form of step previous to each
        perturbation(?) in each iteration.

        This method is intended to be overridden by subclasses to implement
        specific pre-step logic. By default, it does nothing.
        """
        pass


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

    def __init__(self, func: Callable, grad: Callable, func_args: List, grad_args: List, n_red=1):
        super().__init__(func, func_args, n_red)
        self.grad = FuncWrapper(grad, grad_args)


class PowerSeriesGradientPerturbation(GradientPerturbation):
    """
    Objective function perturbation using gradient descent with step size
    reduction according to a power series.
    Has the option to "restart" the power series after a certain number of
    steps.

    func : Callable
        The function to be optimized.
    grad : Callable
        The gradient of the function to be optimized.
    func_args : List, optional
        Additional arguments to be passed to the function, by default [].
    grad_args : List, optional
        Additional arguments to be passed to the gradient function, by default [].
    n_red : int, optional
        The number of reductions, by default 1.
    step_size : float, optional
        The step size for the gradient descent, by default 0.5.
    n_restart : int, optional
        The number of steps after which to restart the power series, by default -1 (no restart).
    """

    def __init__(
        self,
        func: Callable,
        grad: Callable,
        func_args: List = [],
        grad_args: List = [],
        n_red=1,
        step_size=0.5,
        n_restart=-1,
    ):
        super().__init__(func, grad, func_args, grad_args, n_red)
        self.step_size = step_size
        self._l = -1
        self.n_restart = np.inf if n_restart == -1 else n_restart

    def _function_reduction_step(self, x: npt.ArrayLike) -> npt.ArrayLike:
        """
        Perform a function reduction step using gradient descent.

        Parameters
        ----------
        x : npt.ArrayLike
            The current point in the optimization process.

        Returns
        -------
        npt.ArrayLike
            The updated point after performing the reduction step.
        """
        xp = cp if isinstance(x, cp.ndarray) else np
        grad_eval = self.grad(x)
        func_eval = self.func(x)
        loop = True
        while loop:
            self._l += 1
            x_ln = x - self.step_size**self._l * grad_eval / (xp.linalg.norm(grad_eval))
            y_ln = self.func(x_ln)
            if y_ln <= func_eval:
                return x_ln

    def pre_step(self):
        """
        Resets the power series after n steps.

        Returns
        -------
        None
        """
        if not (self._k > 0):
            return
        # possibly restart the power series
        if self._k % self.n_restart == 0:
            self._l = self._k // self.n_restart
