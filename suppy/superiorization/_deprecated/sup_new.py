from typing import Callable
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
from suppy.utils import FuncWrapper
from suppy.utils import ensure_float_array

try:
    import cupy as cp

    NO_GPU = False
except ImportError:
    NO_GPU = True
    cp = np


class BaseSuperiorization(ABC):
    """
    Base class for superiorization algorithms.

    Attributes:
    - basic: The basic algorithm used in the superiorization scheme.

    Methods:
    - __init__(self, basic): Initializes the BaseSuperiorization object.
    - solve(self, x_0): Abstract method for solving the superiorization problem.
    """

    def __init__(self, basic):
        """
        Initializes the BaseSuperiorization object.

        Parameters:
        - basic: The basic object used in the superiorization algorithm.
        """
        self.basic = basic

    @abstractmethod
    def solve(self, x_0):
        """
        Abstract method for solving the superiorization problem.

        Parameters:
        - x0: The initial solution.

        Returns:
        - The final solution.
        """
        pass


class Superiorization(BaseSuperiorization, ABC):
    """
    The Superiorization class represents a superiorization algorithm for
    solving optimization problems.

    Parameters:
    - basic: The basic algorithm used in the superiorization algorithm.
    - func: The objective function to be minimized.
    - func_args: Additional arguments for the objective function.
    - n_red: The number of function reduction steps to perform in each iteration.
    - objective_tol: The tolerance for the change in the objective function value.
    - constr_tol: The tolerance for the change in the proximity function value.

    Attributes:
    - wrapped_func: A wrapper function for the objective function.
    - f_k: The current objective function value.
    - p_k: The current proximity function value.
    - _k: The current iteration counter.
    - all_x_values: An array storing all x values.
    - all_function_values: An array storing all objective function values.
    - all_x_values_function_reduction: An array storing all points achieved via the function reduction step.
    - all_function_values_function_reduction: An array storing all objective function values achieved via the function reduction step.
    - all_x_values_basic: An array storing all points achieved via the basic algorithm.
    - all_function_values_basic: An array storing all objective function values achieved via the basic algorithm.
    """

    def __init__(
        self,
        basic,
        func: Callable,
        func_args=(),
        n_red=1,
        objective_tol: float = 1e-4,
        constr_tol: float = 1e-6,
    ):
        super().__init__(basic)
        self.wrapped_func = FuncWrapper(func, func_args)

        # initialize some variables for the algorithms
        self.f_k = None
        self.p_k = None
        self._k = 0

        self.n_red = n_red
        self.objective_tol = objective_tol
        self.constr_tol = constr_tol

        self.all_x_values = []
        self.all_function_values = []  # array storing all objective function values

        self.all_x_values_function_reduction = (
            []
        )  # array storing all points achieved via the function reduction step
        self.all_function_values_function_reduction = (
            []
        )  # array storing all objective function values achieved via the function reduction step

        self.all_x_values_basic = []  # array storing all points achieved via the basic algorithm
        self.all_function_values_basic = (
            []
        )  # array storing all objective function values achieved via the basic algorithm

    @ensure_float_array
    def solve(self, x_0: npt.NDArray, max_iter: int = 10, storage=False):
        """
        Solve the superiorization problem.

        Parameters:
        - x_0: The initial point for the optimization problem.
        - max_iter: The maximum number of iterations to perform.
        - storage: A boolean indicating whether to store intermediate results or not.

        Returns:
        x: The final solution.
        """

        # initialization of variables
        x = x_0
        self._k = 0  # reset counter if necessary
        stop = False

        # initial function and proximity values
        self.f_k = self.wrapped_func(x_0)
        self.p_k = self.basic.proximity(x_0)

        if storage:
            self._initial_storage(x_0)

        while self._k < max_iter and not stop:
            # perform n_red function reduction steps
            n = 0
            while n < self.n_red:
                x = self._function_reduction_step(x)
                n += 1

            if storage:
                self._storage_function_reduction(x, self.wrapped_func(x))

            # perform basic step
            x = self.basic.step(x)

            if storage:
                self._storage_basic_step(x, self.wrapped_func(x))

            self._k += 1

            # check current function and proximity values
            f_temp = self.wrapped_func(x)
            p_temp = self.basic.proximity(x)

            # enable different stopping criteria for different superiorization algorithms
            stop = self._stopping_criteria(f_temp, p_temp)

            # update function and proximity values
            self.f_k = f_temp
            self.p_k = p_temp

            self._additional_action(x)
        return x

    @abstractmethod
    def _function_reduction_step(self, x):
        pass

    def _stopping_criteria(self, f_temp, p_temp) -> bool:
        """
        Stopping criteria for the superiorization algorithm.

        Parameters:
        - f_temp: The current objective function value.
        - p_temp: The current proximity function value.

        Returns:
        - stop: A boolean indicating whether to stop the algorithm or not.
        """
        stop = (
            abs(f_temp - self.f_k) < self.objective_tol
            and abs(p_temp - self.p_k) < self.constr_tol
        )
        return stop

    def _additional_action(self, x):
        pass

    def _initial_storage(self, x):
        """
        Initialize the storage arrays for storing intermediate results.

        Parameters:
        - x: The initial point for the optimization problem.

        Returns:
        None
        """
        # reset objective values
        self.all_x_values = []
        self.all_function_values = []  # array storing all objective function values

        self.all_x_values_function_reduction = []
        self.all_function_values_function_reduction = []

        self.all_x_values_basic = []
        self.all_function_values_basic = []

        # append initial values
        self.all_x_values.append(x)
        self.all_function_values.append(self.wrapped_func(x))

    def _storage_function_reduction(self, x, f):
        """
        Store intermediate results achieved via the function reduction step.

        Parameters:
        - x: The current point achieved via the function reduction step.
        - f: The current objective function value achieved via the function reduction step.

        Returns:
        None
        """
        self.all_x_values.append(x.copy())
        self.all_function_values.append(f)
        self.all_x_values_function_reduction.append(x.copy())
        self.all_function_values_function_reduction.append(f)

    def _storage_basic_step(self, x, f):
        """
        Store intermediate results achieved via the basic algorithm step.

        Parameters:
        - x: The current point achieved via the basic algorithm step.
        - f: The current objective function value achieved via the basic algorithm step.

        Returns:
        None
        """
        self.all_x_values_basic.append(x.copy())
        self.all_function_values_basic.append(f)
        self.all_x_values.append(x.copy())
        self.all_function_values.append(f)


class GradientSuperiorization(Superiorization, ABC):
    """
    An abstract class representing a base for gradient based superiorization
    algorithms.

    Args:
        basic: The basic algorithm to be improved.
        func (Callable): The objective function to be minimized.
        grad (Callable): The gradient function of the objective function.
        func_args (tuple, optional): Additional arguments for the objective function. Defaults to ().
        grad_args (tuple, optional): Additional arguments for the gradient function. Defaults to ().
        n_red (int, optional): The number of reduction steps to perform. Defaults to 1.
        objective_tol (float, optional): The tolerance for the objective function. Defaults to 1e-4.
        constr_tol (float, optional): The tolerance for the constraints. Defaults to 1e-6.
    """

    def __init__(
        self,
        basic,
        func: Callable,
        grad: Callable,
        func_args=(),
        grad_args=(),
        n_red=1,
        objective_tol: float = 1e-4,
        constr_tol: float = 1e-6,
    ):

        super().__init__(basic, func, func_args, n_red, objective_tol, constr_tol)
        self.wrapped_grad = FuncWrapper(grad, grad_args)


class PowerSeriesGradientSuperiorization(GradientSuperiorization):
    """
    A class representing a gradient based superiorzation algorithm using a
    power decreasing series.

    Args:
        basic: The basic object representing the optimization problem.
        func: The objective function to be minimized.
        grad: The gradient function of the objective function.
        func_args: Additional arguments for the objective function (default: ()).
        grad_args: Additional arguments for the gradient function (default: ()).
        n_red: The number of function reduction steps to perform (default: 1).
        objective_tol: The tolerance for the objective function value (default: 1e-4).
        constr_tol: The tolerance for the constraint violation (default: 1e-6).
        alpha: The step size parameter (default: 0.5).
        n_restart: The number of iterations before resetting the step size parameter (default: -1, indicating no reset).

    Attributes:
        alpha: The step size parameter.
        _l: The current modification parameter for the power series
        n_restart: The number of iterations before resetting the step size parameter.
    """

    def __init__(
        self,
        basic,
        func: Callable,
        grad: Callable,
        func_args=(),
        grad_args=(),
        n_red=1,
        objective_tol: float = 1e-4,
        constr_tol: float = 1e-6,
        alpha=0.5,
        n_restart=-1,
    ):
        super().__init__(basic, func, grad, func_args, grad_args, n_red, objective_tol, constr_tol)
        self.alpha = alpha
        self._l = -1
        if n_restart == -1:
            self.n_restart = np.inf
        else:
            self.n_restart = n_restart

    def _function_reduction_step(self, x):
        """
        Perform the function reduction routine for the power series gradient
        superiorization algorithm.

        Args:
            x: The current solution vector.

        Returns:
            The updated solution vector after performing the function reduction step.
        """
        xp = cp if isinstance(x, cp.ndarray) else np
        # perform the gradient step
        grad = self.wrapped_grad(x)
        y = self.wrapped_func(x)
        loop = True
        while loop:
            self._l += 1

            # calculate temporary results
            x_ln = x - self.alpha ** (self._l) * grad / xp.sqrt((grad**2).sum())
            y_ln = self.wrapped_func(x_ln)

            if y_ln < y:
                return x_ln

            # todo: better else

        grad = self.wrapped_grad(x)
        x = x - self.alpha * grad

        return x

    def _additional_action(self, x):
        if self._k % self.n_restart == 0:
            self._l = self._k // self.n_restart


class SplitSuperiorization(BaseSuperiorization, ABC):
    def __init__(
        self,
        basic,
        input_func: Callable | None = None,
        target_func: Callable | None = None,
        input_func_args=(),
        target_func_args=(),
        n_red=1,
        input_func_tol: float = 1e-4,
        target_func_tol: float = 1e-4,
        constr_tol: float = 1e-6,
    ):

        super().__init__(basic)

        # initialize the input function and gradient
        if input_func is not None:
            self.wrapped_input_func = FuncWrapper(input_func, input_func_args)
        else:
            self.wrapped_input_func = None

        # initialize the target function and gradient
        if target_func is not None:
            self.wrapped_target_func = FuncWrapper(target_func, target_func_args)
        else:
            self.wrapped_target_func = None

        self.input_f_k = None
        self.target_f_k = None
        self.p_k = None
        self._k = 0
        self.n_red = n_red
        self.input_func_tol = input_func_tol
        self.target_func_tol = target_func_tol
        self.constr_tol = constr_tol

    @ensure_float_array
    def solve(self, x_0: npt.NDArray, max_iter: int = 10):
        """
        Solve the superiorization problem.

        Parameters:
        - x_0: The initial point for the optimization problem.
        - max_iter: The maximum number of iterations to perform.

        Returns:
        None
        """

        # initialization of variables
        x = x_0
        self._k = 0  # reset counter if necessary
        stop = False
        y = self.basic.map(x_0)

        # initial function and proximity values
        if self.wrapped_input_func is not None:
            self.input_f_k = self.wrapped_input_func(x_0)
        if self.wrapped_target_func is not None:
            self.target_f_k = self.wrapped_target_func(y)

        self.p_k = self.basic.proximity(x_0, y)

        while self._k < max_iter and not stop:

            # perform n_red function reduction steps
            n = 0
            while n < self.n_red:
                x, y = self._function_reduction_step(x)
                n += 1

            # perform basic step
            x = self.basic.step(x, y)

            self._k += 1

            # check current function and proximity values
            if self.wrapped_input_func is not None:
                input_f_temp = self.wrapped_input_func(x)
            if self.wrapped_target_func is not None:
                target_f_temp = self.wrapped_target_func(self.basic.map(x))
            p_temp = self.basic.proximity(x)

            # check if given stopping criteria are met
            stop = self._stopping_criteria(input_f_temp, target_f_temp, p_temp)

            # update function (if they exist) and proximity values
            if self.wrapped_input_func is not None:
                self.input_f_k = input_f_temp

            if self.wrapped_target_func is not None:
                self.target_f_k = target_f_temp

            self.p_k = p_temp

            self._additional_action(x)

        return x

    @abstractmethod
    def _function_reduction_step(self, x):
        pass

    def _stopping_criteria(self, input_f_temp, target_f_temp, p_temp) -> bool:
        # three stopping criteria have to be met
        input_crit = abs(input_f_temp - self.input_f_k) < self.input_func_tol
        target_crit = abs(target_f_temp - self.target_f_k) < self.target_func_tol
        constr_crit = abs(p_temp - self.p_k) < self.constr_tol
        stop = input_crit and target_crit and constr_crit
        return stop

    def _additional_action(self, x):
        pass


class SplitGradientSuperiorization(SplitSuperiorization, ABC):
    def __init__(
        self,
        basic,
        input_func: Callable | None = None,
        target_func: Callable | None = None,
        input_grad: Callable | None = None,
        target_grad: Callable | None = None,
        input_func_args=(),
        target_func_args=(),
        input_grad_args=(),
        target_grad_args=(),
        n_red=1,
        input_func_tol: float = 1e-4,
        target_func_tol: float = 1e-4,
        constr_tol: float = 1e-6,
    ):

        # make sure that both input func and gradient are given (or neither)
        if (input_func is None and input_grad is not None) or (
            input_func is not None and input_grad is None
        ):
            raise ValueError(
                "Both input function and input gradient have to be given (or neither)."
            )

        if (target_func is None and target_grad is not None) or (
            target_func is not None and target_grad is None
        ):
            raise ValueError(
                "Both target function and target gradient have to be given (or neither)."
            )

        if input_func is None and target_func is None:
            raise ValueError("At least one of the functions has to be given.")

        super().__init__(
            basic,
            input_func,
            target_func,
            input_func_args,
            target_func_args,
            n_red,
            input_func_tol,
            target_func_tol,
            constr_tol,
        )

        if input_grad is not None:
            self.wrapped_input_grad = FuncWrapper(input_grad, input_grad_args)

        if target_grad is not None:
            self.wrapped_target_grad = FuncWrapper(target_grad, target_grad_args)


class PowerSeriesGradientSplitSuperiorization(SplitGradientSuperiorization):
    def __init__(
        self,
        basic,
        input_func: Callable | None = None,
        target_func: Callable | None = None,
        input_grad: Callable | None = None,
        target_grad: Callable | None = None,
        input_func_args=(),
        target_func_args=(),
        input_grad_args=(),
        target_grad_args=(),
        n_red=1,
        input_func_tol: float = 1e-4,
        target_func_tol: float = 1e-4,
        constr_tol: float = 1e-6,
        alpha=0.5,
        n_restart=-1,
    ):

        super().__init__(
            basic,
            input_func,
            target_func,
            input_grad,
            target_grad,
            input_func_args,
            target_func_args,
            input_grad_args,
            target_grad_args,
            input_func_tol,
            target_func_tol,
            constr_tol,
        )
        self.alpha = alpha
        self.n_red = n_red
        self._l = 0
        if n_restart == -1:
            self.n_restart = np.inf

    def _function_reduction_step(self, x):
        """
        Perform the function reduction step for the Power series gradient
        superiorization algorithm.
        """
        xp = cp if isinstance(x, cp.ndarray) else np
        # perform the gradient step
        y = self.basic.map(x)

        if self.wrapped_input_func is not None:
            input_f = self.wrapped_input_func(x)
            input_g = self.wrapped_input_grad(x)

        if self.wrapped_target_func is not None:

            target_f = self.wrapped_target_func(y)
            target_g = self.wrapped_target_grad(y)

        loop = True

        while loop:
            self._l += 1

            # are we moving in the correct direction?
            decreasing = True

            # gradient step in input space
            if self.wrapped_input_func is not None:
                x_ln = x - self.alpha ** (self._l) * input_f / xp.sqrt((input_f**2).sum)
                if self.wrapped_input_func(x_ln) > input_g:
                    decreasing = False
            else:
                x_ln = x

            # gradient step in target space
            if self.wrapped_target_func is not None:
                y_ln = y - self.alpha ** (self._l) * target_f / xp.sqrt((input_f**2).sum)
                if self.wrapped_target_func(y_ln) > target_g:
                    decreasing = False
            else:
                y_ln = y

            if decreasing:  # found a decreasing direction
                return x_ln, y_ln
            else:
                self._l += 1
                # TODO: stopping criteria
