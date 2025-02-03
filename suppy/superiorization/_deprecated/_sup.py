from abc import ABC, abstractmethod
import numpy.typing as npt
from suppy.utils import ensure_float_array
from suppy.perturbations import Perturbation, PowerSeriesGradientPerturbation


class FeasibilityPerturbation(ABC):
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


class Superiorization(FeasibilityPerturbation, ABC):
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
        perturbation_scheme,
        objective_tol: float = 1e-4,
        constr_tol: float = 1e-6,
    ):
        super().__init__(basic)
        self.perturbation_scheme = perturbation_scheme
        self.objective_tol = objective_tol
        self.constr_tol = constr_tol

        # initialize some variables for the algorithms
        self.f_k = None
        self.p_k = None
        self._k = 0

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
        self.f_k = self.perturbation_scheme.func(x_0)
        self.p_k = self.basic.proximity(x_0)

        if storage:
            self._initial_storage(x_0, self.perturbation_scheme.func(x_0))

        while self._k < max_iter and not stop:

            # check if a restart should be performed

            # perform the perturbation schemes update step
            x = self.perturbation_scheme.perturbation_step(x)

            if storage:
                self._storage_function_reduction(x, self.perturbation_scheme.func(x))

            # perform basic step
            x = self.basic.step(x)

            if storage:
                self._storage_basic_step(x, self.perturbation_scheme.func(x))

            self._k += 1

            # check current function and proximity values
            f_temp = self.perturbation_scheme.func(x)
            p_temp = self.basic.proximity(x)

            # enable different stopping criteria for different superiorization algorithms
            stop = self._stopping_criteria(f_temp, p_temp)

            # update function and proximity values
            self.f_k = f_temp
            self.p_k = p_temp

            self._additional_action(x)

        return x

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

    def _initial_storage(self, x, f):
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
        self.all_function_values.append(f)

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


class SplitSuperiorization(FeasibilityPerturbation, ABC):
    def __init__(
        self,
        basic,  # needs to be a split problem
        input_perturbation_scheme: Perturbation | None = None,
        target_perturbation_scheme: Perturbation | None = None,
        input_objective_tol: float = 1e-4,
        target_objective_tol: float = 1e-4,
        constr_tol: float = 1e-6,
    ):
        super().__init__(basic)
        self.input_perturbation_scheme = input_perturbation_scheme
        self.target_perturbation_scheme = target_perturbation_scheme

        self.input_objective_tol = input_objective_tol
        self.target_objective_tol = target_objective_tol
        self.constr_tol = constr_tol

        # initialize some variables for the algorithms
        self.input_f_k = None
        self.target_f_k = None
        self.p_k = None
        self._k = 0

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
        y = self.basic.map(x)
        self._k = 0  # reset counter if necessary
        stop = False

        # initial function and proximity values
        if self.input_perturbation_scheme is not None:
            self.input_f_k = self.input_perturbation_scheme.func(x_0)

        if self.target_perturbation_scheme is not None:
            self.target_f_k = self.target_perturbation_scheme.func(y)

        self.p_k = self.basic.proximity(x_0)

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

    def _stopping_criteria(self, input_f_temp, target_f_temp, p_temp) -> bool:
        """
        Stopping criteria for the superiorization algorithm.

        Parameters:
        - f_temp: The current objective function value.
        - p_temp: The current proximity function value.

        Returns:
        - stop: A boolean indicating whether to stop the algorithm or not.
        """
        input_crit = abs(input_f_temp - self.input_f_k) < self.input_objective_tol
        target_crit = abs(target_f_temp - self.target_f_k) < self.target_objective_tol
        constr_crit = abs(p_temp - self.p_k) < self.constr_tol
        stop = input_crit and target_crit and constr_crit
        return stop

    def _additional_action(self, x, y):
        pass

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
