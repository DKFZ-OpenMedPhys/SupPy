from abc import ABC, abstractmethod
from suppy.projections import Projection
import numpy.typing as npt


class FeasibilityPerturbation(ABC):
    """
    Abstract base class for perturbation approaches of feasibility seeking
    algorithms.

    Parameters
    ----------
    basic : Projection
        The underlying feasibility seeking algorithm.

    Attributes
    ----------
    basic : Projection
        The underlying feasibility seeking algorithm.
    """

    def __init__(self, basic: Projection):
        self.basic = basic

    @abstractmethod
    def solve(self, x_0: npt.NDArray) -> npt.NDArray:
        """
        Solve the perturbed feasibility seeking problem.

        Parameters
        ----------
        x_0 : npt.NDArray
            Initial guess for the solution.

        Returns
        -------
        npt.NDArray
            The superiorized solution.
        """
