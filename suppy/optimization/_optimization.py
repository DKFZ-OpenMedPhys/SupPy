"""Abstract base class for optimization algorithms."""
from abc import ABC, abstractmethod
import numpy.typing as npt


class Optimization(ABC):
    """
    Abstract base class for optimization algorithms.

    Attributes
    ----------
    all_x : list
        Iterates stored during the solve when storage is enabled.
    all_function_values : list
        Objective function values recorded at each iteration.
    t : list
        Cumulative wall-clock time at each iteration.
    t_it : list
        Per-iteration wall-clock times.
    """

    @abstractmethod
    def solve(self, x_0: npt.NDArray, **kwargs) -> npt.NDArray:
        """
        Solve the optimization problem.

        Parameters
        ----------
        x_0 : npt.NDArray
            Initial iterate.

        Returns
        -------
        npt.NDArray
            The solution iterate.
        """
