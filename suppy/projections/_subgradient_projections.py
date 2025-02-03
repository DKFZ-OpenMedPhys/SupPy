from typing import Callable, List
import numpy as np
import numpy.typing as npt

from suppy.projections._projections import BasicProjection

try:
    import cupy as cp

    no_gpu = False
except ImportError:
    no_gpu = True
    cp = np


class SubgradientProjection(BasicProjection):
    """Projection using subgradients."""

    def __init__(
        self,
        func: Callable,
        grad: Callable,
        level: float = 0,
        func_args: List | None = None,
        grad_args: List | None = None,
        relaxation: float = 1,
        idx: npt.NDArray | None = None,
        proximity_flag=True,
        use_gpu=False,
    ):
        """
        Initialize the SubgradientProjection object.

        Parameters:
        - func (Callable): The objective function.
        - grad (Callable): The gradient function.
        - level (float): The level at which to project.
        - func_args (Any): Additional arguments for the objective function.
        - grad_args (Any): Additional arguments for the gradient function.
        - relaxation (float): The relaxation parameter.
        - idx (npt.NDArray | None): The indices to project on.
        - proximity_flag (bool): Flag to use proximity function.
        - use_gpu (bool): Flag to show whether the function and gradient calls are performed on the GPU or not.

        Returns:
        - None
        """
        super().__init__(relaxation, idx, proximity_flag, _use_gpu=use_gpu)
        self.func = func
        self.grad = grad
        self.level = level
        self.func_args = func_args if func_args is not None else []
        self.grad_args = grad_args if grad_args is not None else []

    def func_call(self, x):
        """
        Call the objective function.

        Parameters:
        - x (npt.NDArray): The input array.

        Returns:
        - float: The value of the objective function.
        """
        return self.func(x[self.idx], *self.func_args)

    def grad_call(self, x):
        """
        Call the gradient function.

        Parameters:
        - x (npt.NDArray): The input array.

        Returns:
        - npt.NDArray: The gradient of the objective function.
        """
        return self.grad(x[self.idx], *self.grad_args)

    def _project(self, x: npt.NDArray) -> npt.NDArray:
        """
        Project the input array onto the specified level.

        Parameters:
        - x (npt.NDArray): The input array.

        Returns:
        - npt.NDArray: The projected array.
        """
        xp = cp if isinstance(x, cp.ndarray) else np
        f_x = self.func_call(x)
        g_x = self.grad_call(x)

        if f_x > self.level and xp.linalg.norm(g_x) > 0:
            x[self.idx] -= (f_x - self.level) * g_x / (g_x @ g_x)
        return x

    def level_diff(self, x: npt.NDArray) -> float:
        """
        Calculate the difference between the objective function value and
        the set level.

        Parameters:
        - x (npt.NDArray): The input array.

        Returns:
        - float: The difference between the objective function value and the set level.
        """
        return self.func_call(x) - self.level

    def _proximity(self, x: npt.NDArray, proximity_measures: List) -> float:
        dist = self.level_diff(x)
        dist = dist if dist > 0 else 0
        measures = []
        for measure in proximity_measures:
            if isinstance(measure, tuple):
                if measure[0] == "p_norm":
                    measures.append(dist ** measure[1])
                else:
                    raise ValueError("Invalid proximity measure")
            elif isinstance(measure, str) and measure == "max_norm":
                measures.append(dist)
            else:
                raise ValueError("Invalid proximity measure")
        return measures


class EUDProjection(SubgradientProjection):
    """
    Class representing the EUDProjection.

    This class inherits from the SubgradientProjection class
    and implements the EUD (Equivalent Uniform Dose) projection.

    Parameters:
    - a (float): Exponent used in the EUD projection.
    - level (int): The level of the projection.

    Attributes:
    - a (float): Exponent used in the EUD projection.

    Methods:
    - func_call(x): Computes the EUD projection function.
    - grad_call(x): Computes the gradient of the EUD projection function.
    """

    def __init__(
        self,
        a: float,
        EUD_max: float = 10,
        relaxation: float = 1,
        idx: npt.NDArray | None = None,
        proximity_flag=True,
        use_gpu=False,
    ):
        """Initializes the EUDProjection object."""
        super().__init__(
            self._func,
            self._grad,
            relaxation=relaxation,
            level=EUD_max,
            idx=idx,
            proximity_flag=proximity_flag,
            use_gpu=use_gpu,
        )
        self.a = a

    def _func(self, x):
        """
        Computes the EUD projection function.

        Parameters:
        - x (npt.NDArray): The input array.

        Returns:
        - npt.NDArray: The result of the EUD projection function.
        """
        return (1 / x.shape[0] * ((x**self.a).sum(axis=0))) ** (1 / self.a)

    def _grad(self, x):
        """
        Computes the gradient of the EUD projection function.

        Parameters:
        - x (npt.NDArray): The input array.

        Returns:
        - npt.NDArray: The gradient of the EUD projection function.
        """
        return (
            ((x**self.a).sum()) ** (1 / self.a - 1)
            * (x ** (self.a - 1))
            / len(x) ** (1 / self.a)
        )


class WeightEUDProjection(EUDProjection):
    def __init__(
        self,
        A: npt.NDArray,
        a: float,
        EUD_max: float = 10,
        relaxation: float = 1,
        idx: npt.NDArray | None = None,
        proximity_flag=True,
        use_gpu=False,
    ):
        """Initializes the EUDProjection object."""
        super().__init__(a, EUD_max, idx, proximity_flag=proximity_flag, use_gpu=use_gpu)
        self.A = A

    def func_call(self, x):
        """
        Call the objective function.

        Parameters:
        - x (npt.NDArray): The input array.

        Returns:
        - float: The value of the objective function.
        """
        return self.func(self.A @ x[self.idx], *self.func_args)

    def grad_call(self, x):
        """
        Call the gradient function.

        Parameters:
        - x (npt.NDArray): The input array.

        Returns:
        - npt.NDArray: The gradient of the objective function.
        """
        return (self.A).T @ (self.grad(self.A @ x[self.idx], *self.grad_args))
