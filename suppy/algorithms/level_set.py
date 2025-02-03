from typing import List, Callable
import numpy.typing as npt
from suppy.projections import SequentialProjection, SubgradientProjection
from suppy.projections._projections import Projection
from suppy.utils import ensure_float_array


class LevelSet:
    """Level set scheme that tries to minimize a function in C."""

    def __init__(
        self,
        projections: List[Projection] | SequentialProjection,
        function: Callable,
        gradient: Callable,
        func_args: List | None = None,
        gradient_args: List | None = None,
        save_hist=False,
    ):
        func_args = [] if func_args is None else func_args
        gradient_args = [] if gradient_args is None else gradient_args

        self.func_projection = SubgradientProjection(
            function,
            gradient,
            func_args=func_args,
            grad_args=gradient_args,
            relaxation=1.2,
        )
        # TODO: shold be choosable or simultaneous?
        self.C_projection = SequentialProjection(projections)

        self.save_hist = save_hist
        self.all_x = []
        self.all_y = []

    @ensure_float_array
    def solve(
        self,
        x: npt.NDArray,
        f_init: None | float = None,
        max_iter=1000,
        epsilon=0.1,
        constr_tol=1e-6,
    ):
        """Solve the problem."""
        if self.save_hist is True:
            self.all_x = []
            self.all_x.append(x.copy())
            self.all_y = []
            self.all_y.append(self.func_projection.func_call(x))

        # if an init value is given skip first projection
        if f_init is not None:
            self.func_projection.level = f_init
        else:  # find a feasible point
            x = self.C_projection.solve(x)
            f_x = self.func_projection.func_call(x)
            self.func_projection.level = f_x - max(
                epsilon, abs(f_x) * epsilon
            )  # set the level to the value of the function at the feasible point

            if self.save_hist is True:
                self.all_x.append(x.copy())
                self.all_y.append(f_x)

        converged = False

        while not converged:
            i = 0
            feasible = False
            while i < max_iter and not feasible:
                # x_n = x.copy()

                # #project onto constraint set
                x = self.C_projection.project(x)
                # project onto objective set
                x = self.func_projection.project(x)

                # calculate proximity
                # get current function difference to level set
                level_diff = self.func_projection.level_diff(x)

                # get current constraint proximity
                c_proximity = self.C_projection.proximity(x)

                # level set reached and constraints are satisfied
                if level_diff < 0 and c_proximity < constr_tol:
                    feasible = True

                i += 1

            if i == max_iter:
                converged = True

            else:
                # update level set
                f_x = self.func_projection.func_call(x)
                if self.save_hist is True:
                    self.all_x.append(x.copy())
                    self.all_y.append(f_x)

                # update level set
                self.func_projection.level = f_x - max(epsilon, abs(f_x) * epsilon)

        return x
