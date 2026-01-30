from typing import List, Callable

import numpy as np
import numpy.typing as npt
from suppy.feasibility._bands._ams_extrapolations import AdaptiveStepLandweberHyperslab
from suppy.utils import ensure_float_array
from suppy.perturbations import Perturbation, DummyPerturbation
from ._sup import FeasibilityPerturbation

try:
    import cupy as cp

    NO_GPU = False
except ImportError:
    cp = np
    NO_GPU = True


class SplitSuperiorization(FeasibilityPerturbation):
    """
    A class used to perform split superiorization on a given feasibility
    problem.

    Parameters
    ----------
    basic : object
        An instance of a split problem.
    input_perturbation_scheme : Perturbation or None, optional
        Perturbation scheme for the input, by default None.
    target_perturbation_scheme : Perturbation or None, optional
        Perturbation scheme for the target, by default None.
    input_objective_tol : float, optional
        Tolerance for the input objective function, by default 1e-4.
    target_objective_tol : float, optional
        Tolerance for the target objective function, by default 1e-4.
    prox_tol : float, optional
        Tolerance for the constraint, by default 1e-6.

    Attributes
    ----------
    input_perturbation_scheme : Perturbation or None
        Perturbation scheme for the input.
    target_perturbation_scheme : Perturbation or None
        Perturbation scheme for the target.
    input_objective_tol : float
        Tolerance for the input objective function.
    target_objective_tol : float
        Tolerance for the target objective function.
    prox_tol : float
        Tolerance for the constraint.
    input_f_k : float
        The current objective function value for the input.
    target_f_k : float
        The current objective function value for the target.
    p_k : float
        The current proximity function value.
    _k : int
        The current iteration number.
    all_x_values : list
        Array storing all points achieved via the superiorization algorithm.
    all_function_values : list
        Array storing all objective function values achieved via the superiorization algorithm.
    all_x_values_function_reduction : list
        Array storing all points achieved via the function reduction step.
    all_function_values_function_reduction : list
        Array storing all objective function values achieved via the function reduction step.
    """

    def __init__(
        self,
        basic,  # needs to be a split problem
        input_perturbation_scheme: Perturbation | None = None,
        target_perturbation_scheme: Perturbation | None = None,
    ):
        super().__init__(basic)
        if input_perturbation_scheme is None and target_perturbation_scheme is None:
            raise ValueError(
                "At least one perturbation scheme must be provided for SplitSuperiorization."
            )

        self.input_perturbation_scheme = (
            input_perturbation_scheme
            if input_perturbation_scheme is not None
            else DummyPerturbation()
        )
        self.target_perturbation_scheme = (
            target_perturbation_scheme
            if target_perturbation_scheme is not None
            else DummyPerturbation()
        )

        # initialize some variables for the algorithms
        self.input_f_k = None
        self.target_f_k = None
        self.p_k = None
        self._k = 0

        self.all_x = []
        self.all_function_values = []  # array storing all objective function values
        self.proximities = []  # array storing all proximity function values

        # array storing all points achieved via the function reduction step
        self.all_x_values_function_reduction = []

        self.all_x_function_reduction = []
        self.all_function_values_function_reduction = []
        self.proximities_function_reduction = []

        self.all_x_basic = []
        self.all_function_values_basic = []
        self.proximities_basic = []

    @ensure_float_array
    def solve(
        self,
        x: npt.NDArray,
        max_iter: int = 10,
        prox_tol: float = 1e-6,
        del_prox_tol: float = 1e-8,
        del_prox_n: int = 5,
        proximity_measures: List | None = None,
        del_input_objective_tol: float = 1e-6,
        del_input_objective_n: int = 5,
        del_target_objective_tol: float = 1e-6,
        del_target_objective_n: int = 5,
        storage=False,
        alternative_stopping_criterion: Callable | None = None,
        alternative_stopping_criterion_initial_call: Callable | None = None,
    ) -> np.ndarray:
        """
        Solves the optimization problem using the superiorization method.

        Parameters
        ----------
        x : npt.NDArray
            Starting point for the algorithm.
        max_iter : int, optional
            Maximum number of iterations to perform, by default 500.
        prox_tol : float, optional
            The tolerance for the proximity on the constraints, by default 1e-6.
        del_prox_tol : float, optional
            The tolerance for the change in proximity over the last del_prox_n iterations, by default 1e-8.
        del_prox_n : int, optional
            The number of iterations that del_prox_tol needs to be met in a row, by default 5.
        proximity_measures : List, optional
            The proximity measures to calculate, by default a l2 norm measure is used. Right now only the first in the list is used to check the feasibility.
        del_input_objective_tol
            The tolerance for change in the objective function over the last del_input_objective_n iterations, by default 1e-8.
        del_input_objective_n
            The number of iterations that del_input_objective_tol needs to be met in a row, by default 5.
        del_target_objective_tol
            The tolerance for change in the objective function over the last del_target_objective_n iterations, by default 1e-8.
        del_target_objective_n
            The number of iterations that del_target_objective_tol needs to be met in a row, by default 5.
        storage : bool, optional
            Flag indicating whether to store intermediate solutions, by default False.
        alternative_stopping_criterion : callable, optional
            Alternative stopping criterion
        alternative_stopping_criterion_initial_call : callable, optional
            Initial call for an alternative stopping criterion

        Returns
        -------
        npt.NDArray
            The optimized solution after performing the superiorization method.
        """
        # initialization of variables
        if proximity_measures is None:
            proximity_measures = [("p_norm", 2)]
        else:
            # TODO: check that proximity measures are valid
            _ = None

        self.input_perturbation_scheme.reset()  # reset the input perturbation scheme
        self.target_perturbation_scheme.reset()  # reset the target perturbation scheme

        self._n_tol_input_objective = (
            0  # number of iterations with objective function changes below threshold
        )
        self._n_tol_target_objective = (
            0  # number of iterations with objective function changes below threshold
        )

        self._n_tol_prox = 0  # number of iterations with proximity changes below threshold

        self.t = [0]  # array storing the time for each iteration
        self.l = []

        x_0 = x.copy()

        self._initial_storage(
            x,
            storage,
            [
                self.input_perturbation_scheme.func(x_0),
                self.target_perturbation_scheme.func(self.basic.map(x_0)),
            ],
            self.basic.proximity(x_0, proximity_measures),
        )

        self._k = 0  # reset counter if necessary

        if alternative_stopping_criterion_initial_call is not None:
            stop = alternative_stopping_criterion_initial_call(x, self)
        else:
            stop = False  # criterion for stopping the algorithm

        # initial function and proximity values
        # self.input_f_k = self.input_perturbation_scheme.func(x_0)
        # self.target_f_k = self.target_perturbation_scheme.func(y)

        # self.p_k = self.basic.proximity(x_0, proximity_measures)

        # # if storage:
        # #    self._initial_storage(x_0,self.perturbation_scheme.func(x_0))
        y = None

        while self._k < max_iter and not stop:

            # check if a restart should be performed

            # perform the perturbation schemes update steps and pre steps
            self.input_perturbation_scheme.pre_step(
                x,
                last_proximity=self.proximities[-1][0],
                last_proximity_basic=self.proximities_basic[-1][0],
                last_proximity_function_reduction=self.proximities_function_reduction[-1][0],
                last_function_value=self.all_function_values[-1][0],
                last_function_value_basic=self.all_function_values_basic[-1][0],
            )

            x = self.input_perturbation_scheme.perturbation_step(x)

            self.target_perturbation_scheme.pre_step(
                y,
                last_proximity=self.proximities[-1][1],
                last_proximity_basic=self.proximities_basic[-1][1],
                last_proximity_function_reduction=self.proximities_function_reduction[-1][1],
                last_function_value=self.all_function_values[-1][1],
                last_function_value_basic=self.all_function_values_basic[-1][1],
            )

            y = self.target_perturbation_scheme.perturbation_step(y)

            # post steps
            self.input_perturbation_scheme.post_step(
                x,
                last_proximity=self.proximities[-1][0],
                last_proximity_basic=self.proximities_basic[-1][0],
                last_proximity_function_reduction=self.proximities_function_reduction[-1][0],
                last_function_value=self.all_function_values[-1][0],
                last_function_value_basic=self.all_function_values_basic[-1][0],
            )

            self.target_perturbation_scheme.post_step(
                y,
                last_proximity=self.proximities[-1][1],
                last_proximity_basic=self.proximities_basic[-1][1],
                last_proximity_function_reduction=self.proximities_function_reduction[-1][1],
                last_function_value=self.all_function_values[-1][1],
                last_function_value_basic=self.all_function_values_basic[-1][1],
            )

            self.storage(
                x,
                "function_reduction",
                storage,
                [self.input_perturbation_scheme.func(x), self.target_perturbation_scheme.func(y)],
                self.basic.proximity(x, proximity_measures),
            )

            # perform basic step
            x, y = self.basic.step(x, y)

            self.storage(
                x,
                "basic",
                storage,
                [self.input_perturbation_scheme.func(x), self.target_perturbation_scheme.func(y)],
                self.basic.proximity(x, proximity_measures),
            )

            self._k += 1

            # enable different stopping criteria for different superiorization algorithms
            if alternative_stopping_criterion is not None:
                stop = alternative_stopping_criterion(x, self)
            else:
                stop = self._stopping_criterion(
                    del_input_objective_tol,
                    del_input_objective_n,
                    del_target_objective_tol,
                    del_target_objective_n,
                    prox_tol,
                    del_prox_tol,
                    del_prox_n,
                )

            self._additional_action(x, y)

        self._post_step(x)

        return x

    def _stopping_criterion(
        self,
        del_input_objective_tol: float,
        del_input_objective_n: int,
        del_target_objective_tol: float,
        del_target_objective_n: int,
        prox_tol: float,
        del_prox_tol: float,
        del_prox_n: int,
    ) -> bool:
        """"""
        stop_objective = False  # variable to check if the objective function criteria is met
        stop_objective_input = (
            abs(self.all_function_values[-3][0] - self.all_function_values[-1][0])
            / max(1, self.all_function_values[-3][0])
            < del_input_objective_tol
        )
        stop_objective_target = (
            abs(self.all_function_values[-3][1] - self.all_function_values[-1][1])
            / max(1, self.all_function_values[-3][1])
            < del_target_objective_tol
        )

        if stop_objective_input:
            self._n_tol_input_objective += 1
        else:
            self._n_tol_input_objective = 0

        if stop_objective_target:
            self._n_tol_target_objective += 1

        else:
            self._n_tol_target_objective = 0

        if (self._n_tol_input_objective >= del_input_objective_n) and (
            self._n_tol_target_objective >= del_target_objective_n
        ):  # n objective function changes in input AND output space below threshold
            stop_objective = True

        stop_prox = False  # variable to check if the proximity criteria is met
        # check if proximity values are below the threshold
        if self.proximities[-1][1][0] < prox_tol:  # proximity below goal/tolerance
            stop_prox = True

        # check if the proximity changes are below tolerance level
        if (
            abs(self.proximities[-3][1][0] - self.proximities[-1][1][0])
            / max(1, self.proximities[-3][1][0])
            < del_prox_tol
        ):
            self._n_tol_prox += 1
        else:
            self._n_tol_prox = 0
        if self._n_tol_prox >= del_prox_n:  # n proximity changes below threshold
            stop_prox = True

        # check if both criteria are met
        return stop_objective and stop_prox

    def _additional_action(self, x, y):
        """
        Perform an additional action on the given inputs.

        Parameters
        ----------
        x : type
            Description of parameter `x`.
        y : type
            Description of parameter `y`.

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
        self.proximities = []  # array storing all proximity function values

        self.all_x_function_reduction = []
        self.all_function_values_function_reduction = []
        self.proximities_function_reduction = []

        self.all_x_basic = []
        self.all_function_values_basic = []
        self.proximities_basic = []

        # append initial values
        f_temp = []
        for el in f:
            if not NO_GPU and isinstance(el, cp.ndarray):
                f_temp.append(el.get())
            else:
                f_temp.append(el)

        # modify proximities
        p_temp = []
        for el in p:
            if not NO_GPU and isinstance(el, cp.ndarray):
                p_temp.append(el.get())
            else:
                p_temp.append(el)

        self.all_function_values.append(f_temp)
        self.all_function_values_basic.append(f_temp)
        self.all_function_values_function_reduction.append(f_temp)

        self.proximities.append(p_temp)
        self.proximities_basic.append(p_temp)
        self.proximities_function_reduction.append(p_temp)

        if storage:
            if isinstance(x, np.ndarray):
                self.all_x.append(np.array(x.copy()))
                self.all_x_basic.append(x.copy())
                self.all_x_function_reduction.append(x.copy())

            else:
                self.all_x.append((x.get()))
                self.all_x_basic.append((x.get()))
                self.all_x_function_reduction.append((x.get()))

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
        f_temp = []
        for el in f:
            if not NO_GPU and isinstance(el, cp.ndarray):
                f_temp.append(el.get())
            else:
                f_temp.append(el)
        self.all_function_values.append(f_temp)

        # modify proximities
        p_temp = []
        for el in p:
            if not NO_GPU and isinstance(el, cp.ndarray):
                p_temp.append(el.get())
            else:
                p_temp.append(el)
        self.proximities.append(p_temp)

        if storage and isinstance(x, np.ndarray):
            self.all_x.append(x.copy())
        elif storage:
            self.all_x.append((x.get()))

        if type == "function_reduction":
            self.all_function_values_function_reduction.append(f_temp)
            self.proximities_function_reduction.append(p_temp)

            if storage and isinstance(x, np.ndarray):
                self.all_x_function_reduction.append(x.copy())
            elif storage:
                self.all_x_function_reduction.append((x.get()))

        elif type == "basic":
            self.all_function_values_basic.append(f_temp)
            self.proximities_basic.append(p_temp)

            if storage and isinstance(x, np.ndarray):
                self.all_x_basic.append(x.copy())
            elif storage:
                self.all_x_basic.append((x.get()))

        else:
            raise ValueError("Invalid storage type. Use 'function_reduction' or 'basic'.")

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

        self.all_x = np.array(self.all_x)
        self.all_x_function_reduction = np.array(self.all_x_function_reduction)
        self.all_x_basic = np.array(self.all_x_basic)

        # if isinstance(x, np.ndarray):
        self.all_function_values = np.array(self.all_function_values)
        self.all_function_values_function_reduction = np.array(
            self.all_function_values_function_reduction
        )
        self.all_function_values_basic = np.array(self.all_function_values_basic)
        self.proximities = np.array(self.proximities)
        self.proximities_function_reduction = np.array(self.proximities_function_reduction)
        self.proximities_basic = np.array(self.proximities_basic)
        # else:
        #     # If using cupy, convert all arrays to cupy arrays
        #     # convert all to numpy arrays
        #     self.all_function_values = np.array(
        #         [np.array([el[0].get(),el[1]]) for el in self.all_function_values]
        #     )
        #     self.all_function_values_function_reduction = np.array(
        #         [np.array([el[0].get(),el[1]]) for el in self.all_function_values_function_reduction]
        #     )
        #     self.all_function_values_basic = np.array(
        #         [np.array([el[0].get(),el[1]]) for el in self.all_function_values_basic]
        #     )
        #     self.proximities = np.array([el.get() for el in self.proximities])
        #     self.proximities_function_reduction = np.array(
        #         [el.get() for el in self.proximities_function_reduction]
        #     )
        #     self.proximities_basic = np.array(
        #         [el.get() for el in self.proximities_basic]
        #     )

    # def _initial_storage(self,x,f_input,f_target):
    #     """
    #     Initialize the storage arrays for storing intermediate results.

    #     Parameters:
    #     - x: The initial point for the optimization problem.

    #     Returns:
    #     None
    #     """
    #     #reset objective values
    #     self.all_x_values = []
    #     self.all_function_values = []  # array storing all objective function values

    #     self.all_x_values_function_reduction = []
    #     self.all_function_values_function_reduction = []

    #     self.all_x_values_basic = []
    #     self.all_function_values_basic = []

    #     #append initial values
    #     self.all_x_values.append(x)
    #     self.all_function_values.append(f)

    # def _storage_function_reduction(self,x,f):
    #     """
    #     Store intermediate results achieved via the function reduction step.

    #     Parameters:
    #     - x: The current point achieved via the function reduction step.
    #     - f: The current objective function value achieved via the function reduction step.

    #     Returns:
    #     None
    #     """
    #     self.all_x_values.append(x.copy())
    #     self.all_function_values.append(f)
    #     self.all_x_values_function_reduction.append(x.copy())
    #     self.all_function_values_function_reduction.append(f)

    # def _storage_basic_step(self,x,f):
    #     """
    #     Store intermediate results achieved via the basic algorithm step.

    #     Parameters:
    #     - x: The current point achieved via the basic algorithm step.
    #     - f: The current objective function value achieved via the basic algorithm step.

    #     Returns:
    #     None
    #     """
    #     self.all_x_values_basic.append(x.copy())
    #     self.all_function_values_basic.append(f)
    #     self.all_x_values.append(x.copy())
    #     self.all_function_values.append(f)
