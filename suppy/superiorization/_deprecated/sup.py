from typing import Callable, List
from abc import ABC, abstractmethod

import numpy as np
import numpy.typing as npt
import suppy.projections as pr
from suppy.utils import FuncWrapper
from suppy.utils import ensure_float_array


class BaseSuperiorize(ABC):
    """Base class for superiorization algorithms."""

    def __init__(self, basic):
        """
        Initializes a BaseSuperiorize object.

        Parameters:
        - basic: The basic algorithm to be superiorized.
        """
        self.basic = basic

    @abstractmethod
    def solve(self, x0):
        """
        Abstract method to solve the superiorization problem.

        Parameters:
        - x0: The initial guess for the solution.

        Returns:
        - The solution to the superiorization problem.
        """


class StandardSuperiorize(BaseSuperiorize):
    """Basic superiorization solver for single space with a normal
    function.
    """

    def __init__(
        self,
        basic,
        func: Callable,
        grad: Callable,  # what if gradient free?
        func_args=(),
        grad_args=(),
    ):

        super().__init__(basic)
        self.wrapped_func = FuncWrapper(func, func_args)
        self.wrapped_grad = FuncWrapper(grad, grad_args)

        # TODO: Add keep history option?
        # if keep_history:
        self.all_X_values = []
        self.all_function_values = []  # array storing all objective function values
        self.all_X_values_basic = []  # array storing all points achieved via the basic algorithm
        self.all_function_values_basic = (
            []
        )  # array storing all objective function values achieved via the basic algorithm

        self.all_X_values_gradient = []  # array storing all points achieved via the gradient steps
        self.all_function_values_gradient = (
            []
        )  # array storing all objective function values achieved via the gradient steps

    @ensure_float_array
    def solve(
        self,
        x0: npt.NDArray,
        N_red: int = 1,
        max_iter: int = 10,
        alpha=0.5,
        constr_tol=1e-6,
        objective_tol=1e-4,
        storage=False,
        restart=False,
        N_restart=10,
    ):
        """Solve the superiorization problem."""
        x = x0
        k = 0  # iteration counter
        l = -1  # power series
        stop = False
        # calculate initial values
        f_k = self.wrapped_func(x0)
        p_k = self.basic.proximity(x0)  # check proximity at the initial point

        if storage:
            self.all_X_values.append(x0)
            self.all_function_values.append(self.wrapped_func(x0))

        while k < max_iter and not stop:

            n = 0

            while n < N_red:  # do N_red function reduction steps
                grad = self.wrapped_grad(x)
                y = self.wrapped_func(x)
                loop = True
                while loop:
                    l += 1
                    x_ln = x - alpha ** (l) * grad  # /np.sqrt(np.sum(grad**2))
                    y_ln = self.wrapped_func(x_ln)
                    if y_ln < y:
                        x = x_ln
                        loop = False
                        n += 1
                    else:

                        if l > 10000:
                            loop = False
                            stop = True

                # store results if requested
                if storage:
                    self.all_X_values.append(x.copy())
                    self.all_X_values_gradient.append(x.copy())

                    self.all_function_values.append(y_ln)
                    self.all_function_values_gradient.append(y_ln)

            k += 1
            print("After gradient steps:", y_ln)
            x = self.basic.step(x)
            # check if stopping criterion is met
            f_temp = self.wrapped_func(x)
            print("After basic step:", f_temp)
            p_temp = self.basic.proximity(x)

            # if both objective and constraint are not changing much stop
            if abs(f_temp - f_k) < objective_tol and abs(p_temp - p_k) < constr_tol:
                stop = True
            # update f_k for next step
            f_k = f_temp
            p_k = p_temp

            # store results if requested
            if storage:
                self.all_X_values.append(x.copy())
                self.all_X_values_basic.append(x.copy())

                self.all_function_values.append(self.wrapped_func(x))
                self.all_function_values_basic.append(self.wrapped_func(x))

            if restart is True:  # reset power series indice
                if k % N_restart == 0:
                    l = (
                        k // N_restart
                    )  # for now, should effectively increase l after aeach counter by 1

        # finalize
        if storage:
            self.all_X_values = np.array(self.all_X_values)
            self.all_X_values_basic = np.array(self.all_X_values_basic)
            self.all_X_values_gradient = np.array(self.all_X_values_gradient)
        return x


class SplitSuperiorize(BaseSuperiorize):
    """Superiorization solver for split feasibility problems."""

    def __init__(
        self,
        basic,
        input_func: Callable | None = None,
        input_grad: Callable | None = None,
        target_func: Callable | None = None,
        target_grad: Callable | None = None,
        input_func_args=(),
        input_grad_args=(),
        target_func_args=(),
        target_grad_args=(),
    ):

        super().__init__(basic)
        # TODO: should have better logic!

        self.proximity_values = []  # array storing all proximity values

        self.allX = []  # storing all X values

        self.all_Xbasic_values = []  # array storing all points achieved via the basic algorithm

        # only if function and gradient for input are given instantiate, else throw error
        if (input_func is None and input_grad is not None) or (
            input_func is not None and input_grad is None
        ):
            raise ValueError("Input function and gradient must be given together")

        elif input_func is not None and input_grad is not None:
            self.wrapped_input_func = FuncWrapper(input_func, input_func_args)
            self.wrapped_input_grad = FuncWrapper(input_grad, input_grad_args)

        # same with target:
        if (target_func is None and target_grad is not None) or (
            target_func is not None and target_grad is None
        ):
            raise ValueError("Target function and gradient must be given together")

        elif target_func is not None and target_grad is not None:
            self.wrapped_target_func = FuncWrapper(target_func, target_func_args)
            self.wrapped_target_grad = FuncWrapper(target_grad, target_grad_args)

        # At this point both input and target functions and gradients are either None or well defined functions so checking if one is None should be enough

    def solve(
        self,
        x0,
        N_red=10,
        max_iter=10,
        alpha=0.5,
        restart=False,
        N_restart=10,
        storage=False,
    ):
        # constr_tol = 1e-6,
        # objective_tol = 1e-4,

        x = x0
        k = 0
        l = -1

        stop = False
        self.allX.append(x0)

        # run for k iterations
        while k < max_iter and not stop:

            n = 0

            # perturbation phase
            while n < N_red:  # N function reduction steps
                # calculate function value and gradient for the input space if defined
                if self.wrapped_input_func is not None:
                    input_f = self.wrapped_input_func(x)
                    input_g = self.wrapped_input_grad(x)

                # calculate function value and gradient for the target space if defined
                y = self.basic.map(x)  # current y value

                if self.wrapped_target_func is not None:
                    target_f = self.wrapped_target_func(y)
                    target_g = self.wrapped_target_grad(y)

                loop = True

                while loop:
                    l += 1
                    # gradient step in input space
                    if self.wrapped_input_func is not None:
                        x_ln = x - alpha ** (l) * input_g / np.sqrt(np.sum(input_g**2))
                    else:
                        x_ln = x

                    # gradient step in target space
                    if self.wrapped_target_func is not None:
                        y_ln = y - alpha ** (l) * target_g / np.sqrt(
                            np.sum(target_g**2)
                        )  # gradient step in target space
                    else:
                        y_ln = y

                    decreasing = True
                    # check the different combinations if function values are decreasing

                    if (
                        self.wrapped_input_func is not None
                        and self.wrapped_target_func is not None
                    ):
                        if (
                            self.wrapped_input_func(x_ln) > input_f
                            or self.wrapped_target_func(self.basic.map(x_ln)) > target_f
                        ):
                            decreasing = False

                    elif self.wrapped_input_func is not None:
                        if self.wrapped_input_func(x_ln) > input_f:
                            decreasing = False

                    elif self.wrapped_target_func is not None:
                        if self.wrapped_target_func(self.basic.map(x_ln)) > target_f:
                            decreasing = False

                    if decreasing:  # found a decreasing direction
                        x = x_ln
                        y = y_ln
                        loop = False
                        n += 1
                    else:
                        l += 1
                        if l > 100:  # TODO: This does not kill the n loop
                            # return?
                            loop = False
                            stop = True

            k += 1

            # restart:
            if restart is True:
                if k % N_restart == 0:
                    l = k // N_restart

            x = self.basic.step(x, y)  # step in the basic algorithm
            # TODO: Check that if no function is given y does not change throughout the perturbation phase

        return x


class SubvectorSuperiorize(BaseSuperiorize):
    """Superiorization solver for split feasibility problems."""

    def __init__(
        self,
        basic,
        input_func: Callable | None = None,
        input_grad: Callable | None = None,
        target_funcs: List[Callable] | None = None,
        target_grads: List[Callable] | None = None,
        target_idxs: List[List[int]] | None = None,
        input_func_args=(),
        input_grad_args=(),
        target_func_args=(),
        target_grad_args=(),
    ):

        super().__init__(basic)
        # TODO: should have better logic!

        self.proximity_values = []  # array storing all proximity values

        self.allX = []  # storing all X values

        self.all_Xbasic_values = []  # array storing all points achieved via the basic algorithm

        # only if function and gradient for input are given instantiate, else throw error
        if (input_func is None and input_grad is not None) or (
            input_func is not None and input_grad is None
        ):
            raise ValueError("Input function and gradient must be given together")

        elif input_func is not None and input_grad is not None:
            self.wrapped_input_func = FuncWrapper(input_func, input_func_args)
            self.wrapped_input_grad = FuncWrapper(input_grad, input_grad_args)

        # same with target:
        if (target_funcs is None and target_grads is not None) or (
            target_funcs is not None and target_grads is None
        ):
            raise ValueError("Target function and gradient must be given together")

        elif target_funcs is not None and target_grads is not None:
            # TODO: make sure that func/grad have the same length as args

            # does the function require additional arguments?
            if input_func_args == ():
                self.wrapped_target_funcs = [
                    FuncWrapper(target_func, ()) for target_func in target_funcs
                ]
            else:
                self.wrapped_target_funcs = [
                    FuncWrapper(target_func, target_func_arg)
                    for (target_func, target_func_arg) in zip(target_funcs, target_func_args)
                ]

            # does the gradient require additional arguments?
            if input_grad_args == ():
                self.wrapped_target_grads = [
                    FuncWrapper(target_grad, ()) for target_grad in target_grads
                ]
            else:
                self.wrapped_target_grads = [
                    FuncWrapper(target_grad, target_grad_arg)
                    for (target_grad, target_grad_arg) in zip(target_grads, target_grad_args)
                ]

        if target_idxs is not None:
            self.target_idxs = target_idxs
        else:
            self.target_idxs = [np.s_[:]] * len(target_funcs)
        # At this point both input and target functions and gradients are either None or well defined functions so checking if one is None should be enough

    def solve(
        self,
        x0,
        N_red=10,
        max_iter=10,
        alpha=0.5,
        restart=False,
        N_restart=10,
        storage=False,
    ):
        # constr_tol = 1e-6,
        # objective_tol = 1e-4,

        x = x0
        k = 0
        l = -1

        stop = False
        self.allX.append(x0)

        # run for k iterations
        while k < max_iter and not stop:

            n = 0

            # perturbation phase
            while n < N_red:  # N function reduction steps
                # calculate function value and gradient for the input space if defined
                if self.wrapped_input_func is not None:
                    input_f = self.wrapped_input_func(x)
                    input_g = self.wrapped_input_grad(x)

                # calculate function value and gradient for the target space if defined
                y = self.basic.map(x)  # current y value

                if self.wrapped_target_funcs is not None:  # is a target function given?
                    target_f = np.zeros(len(self.wrapped_target_funcs))
                    target_g = np.zeros(y.shape)
                    for i, (func, grad, idxs) in enumerate(
                        zip(
                            self.wrapped_target_funcs,
                            self.wrapped_target_grads,
                            self.target_idxs,
                        )
                    ):
                        target_f[i] = func(y[idxs])
                        target_g[idxs] = grad(y[idxs])

                loop = True

                while loop:
                    l += 1
                    # gradient step in input space
                    if self.wrapped_input_func is not None:
                        x_ln = x + alpha ** (l) * input_g  # /np.sqrt(np.sum(input_g**2))
                    else:
                        x_ln = x

                    # gradient step in target space
                    if self.wrapped_target_funcs is not None:
                        y_ln = (
                            y + alpha ** (l) * target_g
                        )  # /np.sqrt(np.sum(target_g**2)) # gradient step in target space
                    else:
                        y_ln = y

                    decreasing = True
                    # check the different combinations if function values are decreasing

                    if (
                        self.wrapped_input_func is not None
                        and self.wrapped_target_funcs is not None
                    ):
                        # check if any target function is increasing
                        for i, (func, idxs) in enumerate(
                            zip(self.wrapped_target_funcs, self.target_idxs)
                        ):
                            if func(y[idxs]) > target_f[i]:
                                decreasing = False
                                break

                    elif self.wrapped_input_func is not None:
                        if self.wrapped_input_func(x_ln) > input_f:
                            decreasing = False

                    elif self.wrapped_target_funcs is not None:
                        # check if any target function is increasing
                        for i, (func, idxs) in enumerate(
                            self.wrapped_target_funcs, self.target_idxs
                        ):
                            if func(y[idxs]) > target_f[i]:
                                decreasing = False
                                break

                    if decreasing:  # found a decreasing direction
                        x = x_ln
                        y = y_ln
                        loop = False
                        n += 1
                    else:
                        l += 1
                        if l > 100:  # TODO: This does not kill the n loop
                            # return?
                            loop = False
                            stop = True

            k += 1

            # restart:
            if restart is True:
                if k % N_restart == 0:
                    l = k // N_restart

            x = self.basic.step(x, y)  # step in the basic algorithm
            if storage is True:
                self.allX.append(x)
                self.all_Xbasic_values.append(x)
            # TODO: Check that if no function is given y does not change throughout the perturbation phase

        return x


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import suppy.projections as pr

    def func_1(x):
        return 1 / len(x) * (x**2).sum(axis=0)

    def grad_1(x):
        grad = 1 / len(x) * 2 * x
        return grad / np.sqrt(np.sum(grad**2))

    center_1 = np.array([1.2, 0])
    radius = 1
    center_2 = np.array([0, 1.4])

    # Creating a circle

    Ball_1 = pr.BallProjection(center_1, radius)
    Ball_2 = pr.BallProjection(center_2, radius)
    Proj = pr.SequentialProjection([Ball_1, Ball_2])
    x0 = np.array([2.5, 1.5])
    Test = StandardSuperiorize(Proj, func_1, grad_1)
    x_f = Test.solve(np.array([2.5, 1.5]), 1, 10, storage=True)
