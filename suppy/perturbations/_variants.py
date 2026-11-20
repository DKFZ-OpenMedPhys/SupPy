from suppy.perturbations._base import GradientPerturbation

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


class BarzilaiBorweinPerturbation(GradientPerturbation):
    """
    Objective function perturbation using gradient descent. Each
    perturbation performs a standard function reduction step followed by a step
    with adapted step size according to the Barzilai-Borwein method.

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
    max_restarts : int, optional
        The maximum number of restarts to perform, by default -1 (no limit).
    restart_value : int or Callable[[int], int], optional
        Optional value to reset the power series exponen t on a restart.
        Int -> Will be used every time
        Callable -> Custom values
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
        max_restarts: int = -1,
        restart_value: int | Callable[[int], int] | None = None,
        disable_gradient_scaling: bool = False,
        max_step_size: float = 1e6,
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
        self.max_restarts = np.inf if max_restarts == -1 else max_restarts
        self.restart_value = restart_value
        self._n_restarts = 0
        self.disable_gradient_scaling = disable_gradient_scaling
        self._last_grad_norm = 1
        self.max_step_size = max_step_size

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
            self._remember_func_value(self._known_func_value)  # x unchanged
            return x

        xp = cp if isinstance(x, cp.ndarray) else np
        grad_eval = self.grad(x)
        func_eval = self._initial_func_value(x)

        if grad_eval @ grad_eval <= 0:
            self._remember_func_value(func_eval)  # x unchanged
            return x

        # TODO: Should the storage of last gradient norm be removed?

        grad_norm = 1 if self.disable_gradient_scaling else xp.linalg.norm(grad_eval)
        self._last_grad_norm = grad_norm

        if self.step_size_modifier * self.step_size**self._l * self._last_grad_norm <= 1e-14:
            self._remember_func_value(func_eval)  # x unchanged
            return x

        # grad_norm = xp.linalg.norm(grad_eval) if not self.disable_gradient_scaling else 1

        func_val = 0
        # perform one normal step
        while True:
            self._l += 1
            x_i1 = x - self.step_size_modifier * self.step_size**self._l * grad_eval / grad_norm
            # check that objective is smaller
            func_val = self.func(x_i1)
            if func_val <= func_eval:
                break

        # perform one modified Barzilai-Borwein step
        new_grad = self.grad(x_i1)
        self._l += 1
        s = x_i1 - x
        y = new_grad - grad_eval
        step_size_bb = (s @ s) / (s @ y)
        # alpha_k
        step_size_bb = xp.clip(step_size_bb, 0, self.max_step_size)
        x_bb = x_i1 - self.step_size**self._l * step_size_bb * new_grad
        func_bb = self.func(x_bb)
        if func_bb > func_val:
            print("BB step did not decrease the objective!")

        self._remember_func_value(func_bb)
        return x_bb

    def pre_step(self, x: npt.NDArray, *args, **kwargs) -> None:
        """
        Resets the power series after n steps.

        Parameters
        ----------
        x : npt.NDArray
            Current iterate.
        **kwargs
            Values passed by the superiorization driver, see
            `ObjectivePerturbation.pre_step`.
        """
        super().pre_step(x, *args, **kwargs)
        if self._k <= 0:
            return
        if self._k % self.n_restart == 0 and self._n_restarts < self.max_restarts:
            if self.restart_value is None:
                self._l = self._k // self.n_restart
            elif callable(self.restart_value):
                self._l = self.restart_value(self._k)
            else:
                self._l = self.restart_value
            self._n_restarts += 1

    def reset(self) -> None:
        """Reset the perturbation to its initial state."""
        super().reset()
        self._l = -1
        self._n_restarts = 0
