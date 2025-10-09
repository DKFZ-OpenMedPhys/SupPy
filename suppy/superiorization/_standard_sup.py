"""Normal superiorization algorithm."""
from typing import List
import numpy as np
import time
import numpy.typing as npt
from suppy.utils import ensure_float_array
from suppy.perturbations import Perturbation
from ._sup import FeasibilityPerturbation
from suppy.projections import Projection

try:
    import cupy as cp
except ImportError:
    cp = np


class Superiorization(FeasibilityPerturbation):
    """
    Superiorization algorithm for constrained optimization problems.

    Parameters
    ----------
    basic : Callable
        The underlying feasibility seeking algorithm.
    perturbation_scheme : Perturbation
        The perturbation scheme to be used for superiorization.

    Attributes
    ----------
    basic : Callable
        The underlying feasibility seeking algorithm.
    perturbation_scheme : Perturbation
        The perturbation scheme to be used for superiorization.
    objective_tol : float
        Tolerance for the objective function value change to determine stopping criteria.
    prox_tol : float
        Tolerance for the constraint proximity value change to determine stopping criteria.
    _k : int
        The current iteration number.
    all_x : list | None
        List of all points achieved during the optimization process, only stored if requested by the user.
    all_function_values : list | None
        List of all objective function values achieved during the optimization process, only stored if requested by the user.
    all_x_function_reduction : list | None
        List of all points achieved via the function reduction step, only stored if requested by the user.
    all_function_values_function_reduction : list | None
        List of all objective function values achieved via the function reduction step, only stored if requested by the user.
    all_x_basic : list | None
        List of all points achieved via the basic feasibility seeking algorithm, only stored if requested by the user.
    all_function_values_basic : list | None
        List of all objective function values achieved via the basic feasibility seeking algorithm, only stored if requested by the user.
    """

    def __init__(
        self,
        basic: Projection,
        perturbation_scheme: Perturbation,
    ):

        super().__init__(basic)
        self.perturbation_scheme = perturbation_scheme

        # initialize some variables for the algorithms
        self._k = 0
        self.t = []
        self.t_it = []

        # initialize some arrays for storing of results
        self.all_x = []
        self.all_function_values = []  # array storing all objective function values
        self.all_proximity_values = []  # array storing all proximity function values

        self.all_x_function_reduction = []
        self.all_function_values_function_reduction = []
        self.all_proximity_values_function_reduction = []

        self.all_x_basic = []
        self.all_function_values_basic = []
        self.all_proximity_values_basic = []

    @ensure_float_array
    def solve(
        self,
        x_0: npt.NDArray,
        max_iter: int = 10,
        storage=False,
        prox_tol: float = 1e-6,
        del_prox_tol: float = 1e-8,
        del_prox_n: int = 5,
        proximity_measures: List | None = None,
        del_objective_tol: float = 1e-6,
        del_objective_n: int = 5,
    ) -> npt.NDArray:
        """
        Solve the optimization problem using the superiorization method.

        Parameters
        ----------
        x_0 : npt.NDArray
            Initial guess for the solution.
        max_iter : int, optional
            Maximum number of iterations to perform (default is 10).
        storage : bool, optional
            If True, store intermediate results (default is False).
        prox_tol : float, optional
            Tolerance for the constraint function value to determine stopping criteria, by default 1e-6.
        del_prox_tol : float, optional
            Tolerance for the proximity function value change to determine stopping criteria, by default 1e-8.
        del_prox_n : int, optional
            Number of iterations with proximity changes below the threshold to determine stopping criteria, by default 5.
        proximity_measures : List, optional
            The proximity measures to calculate, by default None. Right now only the first in the list is used to check the feasibility.
        del_objective_tol : float, optional
            Tolernace for the objective function value to determine stopping criteria, by default 1e-6.
        del_objective_n : int, optional
            Number of iterations with objective function changes below the threshold to determine stopping criteria, by default 5.

        Returns
        -------
        npt.NDArray
            The optimized solution.
        """
        if proximity_measures is None:
            proximity_measures = [("p_norm", 2)]
        else:
            # TODO: check that proximity measures are valid
            _ = None
        # initialization of variables

        self.perturbation_scheme.reset()  # reset the perturbation scheme

        self._n_tol_objective = (
            0  # number of iterations with objective function changes below threshold
        )
        self._n_tol_prox = 0  # number of iterations with proximity changes below threshold

        self.l = []

        x = x_0

        # initial function and proximity values

        self._initial_storage(
            x,
            storage,
            self.perturbation_scheme.func(x_0),
            self.basic.proximity(x_0, proximity_measures),
        )

        self._k = 0  # reset counter if necessary
        stop = False

        self.t.append(0)
        t_current = time.time()
        t_init = t_current

        while self._k < max_iter and not stop:
            self.perturbation_scheme.pre_step(
                x,
                last_proximity=self.all_proximity_values[-1],
                last_proximity_basic=self.all_proximity_values_basic[-1],
                last_proximity_function_reduction=self.all_proximity_values_function_reduction[-1],
                last_function_value=self.all_function_values[-1],
                last_function_value_basic=self.all_function_values_basic[-1],
            )  # perform any pre-step actions of the perturbation scheme

            # perform the perturbation schemes update step
            x = self.perturbation_scheme.perturbation_step(x)
            self.storage(
                x,
                "function_reduction",
                storage,
                self.perturbation_scheme.func(x),
                self.basic.proximity(x, proximity_measures),
            )

            self.perturbation_scheme.post_step(
                x,
                last_proximity=self.all_proximity_values[-1],
                last_proximity_basic=self.all_proximity_values_basic[-1],
                last_proximity_function_reduction=self.all_proximity_values_function_reduction[-1],
                last_function_value=self.all_function_values[-1],
                last_function_value_basic=self.all_function_values_basic[-1],
            )  # perform any post-step actions of the perturbation scheme

            # perform basic step
            x = self.basic.step(x)

            # check current function and proximity values

            self.storage(
                x,
                "basic",
                storage,
                self.perturbation_scheme.func(x),
                self.basic.proximity(x, proximity_measures),
            )

            self._k += 1

            # self.l.append(self.perturbation_scheme._l)

            # enable different stopping criteria for different superiorization algorithms
            stop = self._stopping_criterion(
                del_objective_tol, del_objective_n, prox_tol, del_prox_tol, del_prox_n
            )

            self._additional_action(x)
            self.t.append(time.time() - t_init)
            self.t_it.append(time.time() - t_current)
            t_current = time.time()
        self._post_step(x)

        return x

    def _stopping_criterion(
        self,
        del_objective_tol: float,
        del_objective_n: int,
        prox_tol: float,
        del_prox_tol: float,
        del_prox_n: int,
    ) -> bool:
        """
        Determine if the stopping criterion for the optimization process are
        met.

        Parameters
        ----------
        f_temp : float
            The current value of the objective function.
        p_temp : List[float]
            The current proximity values to the constraints.
        objective_tol : float
            Tolerance for the objective function value change to determine stopping criteria.
        prox_tol : float
            Tolerance for the constraint proximity value change to determine stopping criteria.

        Returns
        -------
        bool
            True if the stopping criteria are met, False otherwise.
        """

        stop_objective = False  # variable to check if the objective function criteria is met
        # check objective function criteria f_(k+1)-f_k<= (delta f)* is met in last n_tol_objective iterations
        if (
            abs(self.all_function_values[-3] - self.all_function_values[-1])
            / max(1, self.all_function_values[-3])
            < del_objective_tol
        ):
            self._n_tol_objective += 1
        else:
            self._n_tol_objective = 0
        if (
            self._n_tol_objective >= del_objective_n
        ):  # n objective function changes below threshold
            stop_objective = True

        stop_prox = False  # variable to check if the proximity criteria is met
        # check if proximity values are below the threshold
        if self.all_proximity_values[-1][0] < prox_tol:  # proximity below goal/tolerance
            stop_prox = True

        # check if the proximity changes are below tolerance level
        if (
            abs(self.all_proximity_values[-3][0] - self.all_proximity_values[-1][0])
            / max(1, self.all_proximity_values[-3][0])
            < del_prox_tol
        ):
            self._n_tol_prox += 1
        else:
            self._n_tol_prox = 0
        if self._n_tol_prox >= del_prox_n:  # n proximity changes below threshold
            stop_prox = True

        # check if both criteria are met
        return stop_objective and stop_prox

    def _additional_action(self, x: npt.NDArray):
        """
        Perform an additional action on the input, in case it is needed.

        Parameters
        ----------
        x : npt.NDArray
            The current iterate

        Returns
        -------
        None
        """

    def _initial_storage(self, x, storage, f, p):
        """
        Initializes storage for objective values and appends initial values.

        Parameters
        ----------
        x : array-like
            Initial values of the variables.
        f : array-like
            Initial values of the objective function.
        p : array-like
            Proximity function value
        """
        # reset objective values
        self.all_x = []
        self.all_function_values = []  # array storing all objective function values
        self.all_proximity_values = []  # array storing all proximity function values

        self.all_x_function_reduction = []
        self.all_function_values_function_reduction = []
        self.all_proximity_values_function_reduction = []

        self.all_x_basic = []
        self.all_function_values_basic = []
        self.all_proximity_values_basic = []

        # append initial values
        self.all_function_values.append(f)
        self.all_function_values_basic.append(f)
        self.all_function_values_function_reduction.append(f)

        self.all_proximity_values.append(p)
        self.all_proximity_values_basic.append(p)
        self.all_proximity_values_function_reduction.append(p)

        if storage:
            self.all_x.append(x.copy())
            self.all_x_basic.append(x.copy())
            self.all_x_function_reduction.append(x.copy())

    def storage(
        self, x: npt.NDArray, type: str, storage: bool = True, f: float = None, p: float = None
    ):
        """
        Stores the given values of x and f into the corresponding lists.

        Parameters
        ----------
        x : npt.NDArray
            The current value of the variable x to be stored.
        type : str
            The type of storage to be used, either "function_reduction" or "basic".
        storage : bool, optional
            If True, store the values of x
        """

        # always store all function and proximity values
        self.all_function_values.append(f)
        self.all_proximity_values.append(p)
        if storage:
            self.all_x.append(x.copy())

        if type == "function_reduction":
            self.all_function_values_function_reduction.append(f)
            self.all_proximity_values_function_reduction.append(p)
            if storage:
                self.all_x_function_reduction.append(x.copy())

        elif type == "basic":
            self.all_function_values_basic.append(f)
            self.all_proximity_values_basic.append(p)

            if storage:
                self.all_x_basic.append(x.copy())
        else:
            raise ValueError("Invalid storage type. Use 'function_reduction' or 'basic'.")

    # def _storage_function_reduction(self, x: npt.NDArray, f: float, p: float):
    #     """
    #     Stores the given values of x and f into the corresponding lists.

    #     Parameters
    #     ----------
    #     x : npt.NDArray
    #         The current value of the variable x to be stored.
    #     f : float
    #         The current value of the function f to be stored.
    #     p : float
    #         The current value of the proximity function p to be stored.

    #     Notes
    #     -----
    #     This method appends the given values of x and f to the lists
    #     `all_x`, `all_function_values`, `all_x_function_reduction`,
    #     and `all_function_values_function_reduction`.
    #     """
    #     self.all_x_function_reduction.append(x.copy())
    #     self.all_function_values_function_reduction.append(f)
    #     self.all_proximity_values_function_reduction.append(p)

    # def _storage_basic_step(self, x: npt.NDArray, f: float, p: float):
    #     """
    #     Stores the current values of x and f in the respective lists.

    #     Parameters
    #     ----------
    #     x : array-like
    #         The current value of the variable x.
    #     f : float
    #         The current value of the function f.
    #     p : float
    #         The current value of the proximity function p.

    #     Notes
    #     -----
    #     This method appends the current values of x and f to both the basic and
    #     general lists of x values and function values.
    #     """
    #     self.all_x_basic.append(x.copy())
    #     self.all_function_values_basic.append(f)
    #     self.all_x.append(x.copy())
    #     self.all_function_values.append(f)
    #     self.all_proximity_values_basic.append(p)
    #     self.all_proximity_values.append(p)

    def _post_step(self, x: npt.NDArray):
        """
        Perform an action after the optimization process has finished.

        Parameters
        ----------
        x : array-like
            The current value of the variable x.

        Returns
        -------
        None
        """
        if isinstance(x, np.ndarray):
            self.all_x = np.array(self.all_x)
            self.all_function_values = np.array(self.all_function_values)
            self.all_x_function_reduction = np.array(self.all_x_function_reduction)
            self.all_function_values_function_reduction = np.array(
                self.all_function_values_function_reduction
            )
            self.all_x_basic = np.array(self.all_x_basic)
            self.all_function_values_basic = np.array(self.all_function_values_basic)
            self.all_proximity_values = np.array(self.all_proximity_values)
            self.all_proximity_values_function_reduction = np.array(
                self.all_proximity_values_function_reduction
            )
            self.all_proximity_values_basic = np.array(self.all_proximity_values_basic)
        else:
            # If using cupy, convert all arrays to cupy arrays
            # convert all to numpy arrays
            self.all_x = np.array([el.get() for el in self.all_x])
            self.all_function_values = np.array([el.get() for el in self.all_function_values])
            self.all_x_function_reduction = np.array(
                [el.get() for el in self.all_x_function_reduction]
            )
            self.all_function_values_function_reduction = np.array(
                [el.get() for el in self.all_function_values_function_reduction]
            )
            self.all_x_basic = np.array([el.get() for el in self.all_x_basic])
            self.all_function_values_basic = np.array(
                [el.get() for el in self.all_function_values_basic]
            )
            self.all_proximity_values = np.array([el.get() for el in self.all_proximity_values])
            self.all_proximity_values_function_reduction = np.array(
                [el.get() for el in self.all_proximity_values_function_reduction]
            )
            self.all_proximity_values_basic = np.array(
                [el.get() for el in self.all_proximity_values_basic]
            )

        # xp = cp if isinstance(x, cp.ndarray) else np
        # self.all_function_values = xp.array(self.all_function_values)
        # self.all_x_function_reduction = xp.array(self.all_x_function_reduction)
        # self.all_function_values_function_reduction = xp.array(
        #     self.all_function_values_function_reduction
        # )
        # self.all_x_basic = xp.array(self.all_x_basic)
        # self.all_function_values_basic = xp.array(self.all_function_values_basic)
        # self.all_proximity_values = xp.array(self.all_proximity_values)
        # self.all_proximity_values_function_reduction = xp.array(
        #     self.all_proximity_values_function_reduction
        # )
        # self.all_proximity_values_basic = xp.array(self.all_proximity_values_basic)
