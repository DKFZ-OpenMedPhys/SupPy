from typing import List
import numpy.typing as npt
from suppy.utils import ensure_float_array
from suppy.perturbations import Perturbation
from ._sup import FeasibilityPerturbation


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
    constr_tol : float, optional
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
    constr_tol : float
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
        self.input_perturbation_scheme = input_perturbation_scheme
        self.target_perturbation_scheme = target_perturbation_scheme

        # initialize some variables for the algorithms
        self.input_f_k = None
        self.target_f_k = None
        self.p_k = None
        self._k = 0

        self.all_x_values = []
        self.all_function_values = []  # array storing all objective function values

        # array storing all points achieved via the function reduction step
        self.all_x_values_function_reduction = []

        # array storing all objective function values achieved via the function reduction step
        self.all_function_values_function_reduction = []

        # array storing all points achieved via the basic algorithm
        self.all_x_values_basic = []

        # array storing all objective function values achieved via the basic algorithm
        self.all_function_values_basic = []

    @ensure_float_array
    def solve(
        self,
        x_0: npt.NDArray,
        max_iter: int = 10,
        storage=False,
        input_objective_tol: float = 1e-6,
        target_objective_tol: float = 1e-6,
        constr_tol: float = 1e-6,
        proximity_measures: List | None = None,
    ) -> npt.NDArray:
        """
        Solves the optimization problem using the superiorization method.

        Parameters
        ----------
        x_0 : npt.NDArray
            Initial guess for the solution.
        max_iter : int, optional
            Maximum number of iterations to perform (default is 10).
        storage : bool, optional
            If True, stores intermediate results (default is False).

        Returns
        -------
        npt.NDArray
            The optimized solution after performing the superiorization method.
        """
        # initialization of variables
        x = x_0
        y = self.basic.map(x)
        self._k = 0  # reset counter if necessary
        stop = False

        # initial function and proximity values
        if self.input_perturbation_scheme is not None:
            self.input_f_k = self.input_perturbation_scheme.func(x_0)

        if self.target_perturbation_scheme is not None:
            self.target_f_k = self.target_perturbation_scheme.func(y)

        self.p_k = self.basic.proximity(x_0, proximity_measures)

        # if storage:
        #    self._initial_storage(x_0,self.perturbation_scheme.func(x_0))

        while self._k < max_iter and not stop:

            # check if a restart should be performed

            # perform the perturbation schemes update step
            if self.input_perturbation_scheme is not None:
                x = self.input_perturbation_scheme.perturbation_step(x)

            if self.target_perturbation_scheme is not None:
                y = self.target_perturbation_scheme.perturbation_step(y)

            # if storage:
            #    self._storage_function_reduction(x,self.perturbation_scheme.func(x))

            # perform basic step
            x, y = self.basic.step(x, y)

            # if storage:
            #    self._storage_basic_step(x,self.perturbation_scheme.func(x))

            # check current function and proximity values

            if self.input_perturbation_scheme is not None:
                input_f_temp = self.input_perturbation_scheme.func(x)

            if self.target_perturbation_scheme is not None:
                target_f_temp = self.target_perturbation_scheme.func(y)

            p_temp = self.basic.proximity(x, y)

            # enable different stopping criteria for different superiorization algorithms
            stop = self._stopping_criteria(input_f_temp, target_f_temp, p_temp)

            # update function and proximity values
            if self.input_perturbation_scheme is not None:
                self.input_f_k = input_f_temp

            if self.target_perturbation_scheme is not None:
                self.target_f_k = target_f_temp

            self.p_k = p_temp

            self._additional_action(x, y)

            self._k += 1

        return x

    def _stopping_criteria(
        self,
        input_f_temp,
        target_f_temp,
        p_temp,
        input_objective_tol,
        target_objective_tol,
        constr_tol,
    ) -> bool:
        """
        Check if the stopping criteria for the optimization process are met.

        Parameters
        ----------
        input_f_temp : float
            The current value of the input objective function.
        target_f_temp : float
            The current value of the target objective function.
        p_temp : float
            The current value of the constraint parameter.
        input_objective_tol : float
            Tolerance for the input objective function.
        target_objective_tol : float
            Tolerance for the target objective function.
        constr_tol : float
            Tolerance for the constraint.

        Returns
        -------
        bool
            True if all stopping criteria are met, False otherwise.
        """

        input_crit = abs(input_f_temp - self.input_f_k) < input_objective_tol
        target_crit = abs(target_f_temp - self.target_f_k) < target_objective_tol

        constr_crit = p_temp < constr_tol
        stop = input_crit and target_crit and constr_crit
        return stop

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
