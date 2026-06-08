"""FISTA optimization algorithms."""
from typing import Callable, List
import warnings
import numpy as np
import time
import numpy.typing as npt
from suppy.utils import ensure_float_array, FuncWrapper
from ._optimization import Optimization

try:
    import cupy as cp

    NO_GPU = False
except ImportError:
    cp = np
    NO_GPU = True


class FISTA(Optimization):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for composite
    optimization problems of the form:

        min_x  f(x) + g(x)

    where f has an L-Lipschitz continuous gradient and g has a tractable
    proximal operator (e.g. a projection onto a convex constraint set).

    Parameters
    ----------
    func : Callable
        The smooth objective function f.
    grad : Callable
        The gradient of f.
    project : Callable
        Proximal operator of g, or projection onto the constraint set.
        Called as ``project(x, *project_args)``.
    func_args : List, optional
        Additional arguments bound to func, by default [].
    grad_args : List, optional
        Additional arguments bound to grad, by default [].
    project_args : List, optional
        Additional arguments bound to project, by default [].
    step_size : float, optional
        Step size 1/L where L is the Lipschitz constant of grad f,
        by default 1.0.

    Attributes
    ----------
    func : FuncWrapper
        Wrapped objective function.
    grad : FuncWrapper
        Wrapped gradient function.
    project : FuncWrapper
        Wrapped proximal / projection operator.
    step_size : float
        Step size used for gradient steps.
    _k : int
        Current iteration count.
    all_x : list
        Iterates stored when storage is enabled.
    all_function_values : list
        Objective function value f(x_k) at every iteration.
    all_iterate_changes : list
        Euclidean norm ||x_k - x_{k-1}|| at every iteration.
    t : list
        Cumulative wall-clock time at each iteration.
    t_it : list
        Per-iteration wall-clock times.

    References
    ----------
    - [1] Beck, A., & Teboulle, M. (2009). A fast iterative
      shrinkage-thresholding algorithm for linear inverse problems.
      SIAM journal on imaging sciences, 2(1), 183-202.
    """

    def __init__(
        self,
        func: Callable,
        grad: Callable,
        project: Callable,
        func_args: List | None = None,
        grad_args: List | None = None,
        project_args: List | None = None,
        step_size: float = 1.0,
    ):
        if func_args is None:
            func_args = []
        if grad_args is None:
            grad_args = []
        if project_args is None:
            project_args = []

        self.func = FuncWrapper(func, func_args)
        self.grad = FuncWrapper(grad, grad_args)
        self.project = FuncWrapper(project, project_args)
        self.step_size = step_size

        self._k = 0
        self.t = []
        self.t_it = []
        self.all_x = []
        self.all_function_values = []
        self.all_iterate_changes = []

    @ensure_float_array
    def solve(
        self,
        x_0: npt.NDArray,
        max_iter: int = 100,
        del_objective_tol: float = 1e-6,
        del_objective_n: int = 5,
        del_x_tol: float = 1e-8,
        del_x_n: int = 5,
        storage: bool = False,
        storage_iters: List[int] | int | None = None,
        alternative_stopping_criterion: Callable | None = None,
        alternative_stopping_criterion_initial_call: Callable | None = None,
    ) -> np.ndarray:
        """
        Run FISTA to solve the optimization problem.

        Parameters
        ----------
        x_0 : npt.NDArray
            Initial iterate.
        max_iter : int, optional
            Maximum number of iterations, by default 100.
        del_objective_tol : float, optional
            Relative tolerance on the objective change for stopping,
            by default 1e-6.
        del_objective_n : int, optional
            Number of consecutive iterations below del_objective_tol
            required to trigger stopping, by default 5.
        del_x_tol : float, optional
            Absolute tolerance on the iterate change ||x_k - x_{k-1}||
            for stopping, by default 1e-8.
        del_x_n : int, optional
            Number of consecutive iterations below del_x_tol required
            to trigger stopping, by default 5.
        storage : bool, optional
            If True, store iterates in all_x, by default False.
        storage_iters : List[int] or int or None, optional
            Controls which iterates are stored when storage=True.
            None stores every iteration; an int n stores every n-th;
            a list stores exactly those indices (0 = initial point).
        alternative_stopping_criterion : Callable, optional
            Custom stopping criterion called as ``criterion(x, self) -> bool``.
        alternative_stopping_criterion_initial_call : Callable, optional
            Called once before the main loop as ``criterion(x, self) -> bool``.

        Returns
        -------
        npt.NDArray
            The solution iterate.
        """

        def _should_store(idx):
            if storage_iters is None:
                return True
            if isinstance(storage_iters, int):
                return idx % storage_iters == 0
            return idx in storage_iters

        self._k = 0
        self._n_tol_objective = 0
        self._n_tol_x = 0

        x = x_0
        y = x_0.copy()
        t_momentum = 1.0

        self._initial_storage(x, storage and _should_store(0), self.func(x), 0.0)

        if alternative_stopping_criterion_initial_call is not None:
            stop = alternative_stopping_criterion_initial_call(x, self)
        else:
            stop = False

        self.t.append(0)
        t_current = time.time()
        t_init = t_current

        while self._k < max_iter and not stop:
            # gradient step at the momentum point, then apply proximal operator
            x_new = self.project(y - self.step_size * self.grad(y))

            # FISTA momentum update
            t_new = (1.0 + (np.sqrt(1.0 + 4.0 * t_momentum**2))) / 2.0

            xp = cp if not isinstance(x_new, np.ndarray) else np
            delta_x = xp.linalg.norm(x_new - x)

            y = x_new + (t_momentum - 1.0) / t_new * (x_new - x)
            x = x_new
            t_momentum = t_new

            self._k += 1

            self.storage(
                x,
                storage=storage and _should_store(self._k),
                f=self.func(x),
                delta_x=delta_x,
            )

            if alternative_stopping_criterion is not None:
                stop = alternative_stopping_criterion(x, self)
            else:
                stop = self._stopping_criterion(
                    del_objective_tol, del_objective_n, del_x_tol, del_x_n
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
        del_x_tol: float,
        del_x_n: int,
    ) -> bool:
        """
        Check whether the objective-change and iterate-change stopping
        criteria are simultaneously satisfied.

        Parameters
        ----------
        del_objective_tol : float
            Relative tolerance on objective change.
        del_objective_n : int
            Number of consecutive iterations below del_objective_tol to stop.
        del_x_tol : float
            Absolute tolerance on iterate change.
        del_x_n : int
            Number of consecutive iterations below del_x_tol to stop.

        Returns
        -------
        bool
            True when both criteria are satisfied.
        """
        if len(self.all_function_values) < 2:
            return False

        if (
            abs(self.all_function_values[-2] - self.all_function_values[-1])
            / max(1, abs(self.all_function_values[-2]))
            < del_objective_tol
        ):
            self._n_tol_objective += 1
        else:
            self._n_tol_objective = 0
        stop_objective = self._n_tol_objective >= del_objective_n

        if self.all_iterate_changes[-1] < del_x_tol:
            self._n_tol_x += 1
        else:
            self._n_tol_x = 0
        stop_x = self._n_tol_x >= del_x_n

        return stop_objective and stop_x

    def _additional_action(self, x: npt.NDArray) -> None:
        """Hook for subclasses to perform extra work each iteration."""

    def _initial_storage(self, x: npt.NDArray, storage: bool, f: float, delta_x: float) -> None:
        """
        Reset all storage lists and record the initial state.

        Parameters
        ----------
        x : npt.NDArray
            Initial iterate.
        storage : bool
            Whether to store x in all_x.
        f : float
            Initial objective value.
        delta_x : float
            Initial iterate change (always 0).
        """
        self.all_x = []
        self.all_function_values = []
        self.all_iterate_changes = []
        self.t = []
        self.t_it = []

        self.all_function_values.append(f)
        self.all_iterate_changes.append(delta_x)

        if storage:
            if isinstance(x, np.ndarray):
                self.all_x.append(x.copy())
            else:
                self.all_x.append(x.get())

    def storage(
        self,
        x: npt.NDArray,
        storage: bool = True,
        f: float | None = None,
        delta_x: float = 0.0,
    ) -> None:
        """
        Record per-iteration quantities.

        Parameters
        ----------
        x : npt.NDArray
            Current iterate.
        storage : bool, optional
            If True, append x to all_x.
        f : float, optional
            Objective value f(x_k).
        delta_x : float, optional
            Iterate change ||x_k - x_{k-1}||.
        """
        self.all_function_values.append(f)
        self.all_iterate_changes.append(delta_x)

        if storage:
            if isinstance(x, np.ndarray):
                self.all_x.append(x.copy())
            else:
                self.all_x.append(x.get())

    def _post_step(self, x: npt.NDArray) -> None:
        """
        Convert stored lists to numpy arrays after the solve completes.

        Parameters
        ----------
        x : npt.NDArray
            Final iterate (used to determine CPU vs GPU path).
        """
        xp = cp if not isinstance(x, np.ndarray) else np
        self.all_x = xp.array(self.all_x)
        if isinstance(x, np.ndarray):
            self.all_function_values = np.array(self.all_function_values)
        else:
            self.all_function_values = np.array([el.get() for el in self.all_function_values])


class FISTABacktracking(FISTA):
    """
    FISTA with backtracking line search to adaptively estimate the Lipschitz
    constant of grad f.

    At each iteration the gradient at the momentum point y_k is computed once.
    A candidate step size 1/L is then tested against the descent lemma:

        f(x_new) ≤ f(y_k) + <grad_f(y_k), x_new − y_k> + (L/2) ||x_new − y_k||²

    If the condition is violated, L is inflated by eta and the proximal step is
    repeated.  The estimate L warm-starts the next iteration, so the inner loop
    is typically free once L has stabilised.

    Parameters
    ----------
    func : Callable
        The smooth objective function f.
    grad : Callable
        The gradient of f.
    project : Callable
        Proximal operator of g, or projection onto the constraint set.
        Called as ``project(x, *project_args)``.
    func_args : List, optional
        Additional arguments bound to func, by default [].
    grad_args : List, optional
        Additional arguments bound to grad, by default [].
    project_args : List, optional
        Additional arguments bound to project, by default [].
    step_size : float, optional
        Initial step size 1/L_0 used as a warm start for the first
        backtracking search, by default 1.0.
    eta : float, optional
        Inflation factor applied to L when the descent lemma is violated,
        must be > 1, by default 2.0.
    max_backtrack : int, optional
        Maximum number of inflation steps per iteration. A warning is issued
        if reached, by default 50.

    Attributes
    ----------
    L : float
        Current Lipschitz estimate; updated after every iteration.
    eta : float
        Inflation factor.
    all_step_sizes : list or np.ndarray
        Effective step size 1/L used at each iteration (including iteration 0).

    References
    ----------
    - [1] Beck, A., & Teboulle, M. (2009). A fast iterative
      shrinkage-thresholding algorithm for linear inverse problems.
      SIAM journal on imaging sciences, 2(1), 183-202. (Algorithm 2)
    """

    def __init__(
        self,
        func: Callable,
        grad: Callable,
        project: Callable,
        func_args: List | None = None,
        grad_args: List | None = None,
        project_args: List | None = None,
        step_size: float = 1.0,
        eta: float = 2.0,
        max_backtrack: int = 50,
    ):
        if eta <= 1.0:
            raise ValueError("eta must be greater than 1.")
        super().__init__(func, grad, project, func_args, grad_args, project_args, step_size)
        self.eta = eta
        self.max_backtrack = max_backtrack
        self.L = 1.0 / step_size
        self.all_step_sizes: list = []

    @ensure_float_array
    def solve(
        self,
        x_0: npt.NDArray,
        max_iter: int = 100,
        del_objective_tol: float = 1e-6,
        del_objective_n: int = 5,
        del_x_tol: float = 1e-8,
        del_x_n: int = 5,
        storage: bool = False,
        storage_iters: List[int] | int | None = None,
        alternative_stopping_criterion: Callable | None = None,
        alternative_stopping_criterion_initial_call: Callable | None = None,
    ) -> np.ndarray:
        """
        Run FISTA with backtracking to solve the optimization problem.

        Parameters
        ----------
        x_0 : npt.NDArray
            Initial iterate.
        max_iter : int, optional
            Maximum number of iterations, by default 100.
        del_objective_tol : float, optional
            Relative tolerance on the objective change for stopping,
            by default 1e-6.
        del_objective_n : int, optional
            Number of consecutive iterations below del_objective_tol
            required to trigger stopping, by default 5.
        del_x_tol : float, optional
            Absolute tolerance on the iterate change ||x_k - x_{k-1}||
            for stopping, by default 1e-8.
        del_x_n : int, optional
            Number of consecutive iterations below del_x_tol required
            to trigger stopping, by default 5.
        storage : bool, optional
            If True, store iterates in all_x, by default False.
        storage_iters : List[int] or int or None, optional
            Controls which iterates are stored when storage=True.
            None stores every iteration; an int n stores every n-th;
            a list stores exactly those indices (0 = initial point).
        alternative_stopping_criterion : Callable, optional
            Custom stopping criterion called as ``criterion(x, self) -> bool``.
        alternative_stopping_criterion_initial_call : Callable, optional
            Called once before the main loop as ``criterion(x, self) -> bool``.

        Returns
        -------
        np.ndarray
            The solution iterate.
        """

        def _should_store(idx):
            if storage_iters is None:
                return True
            if isinstance(storage_iters, int):
                return idx % storage_iters == 0
            return idx in storage_iters

        self._k = 0
        self._n_tol_objective = 0
        self._n_tol_x = 0
        self.L = 1.0 / self.step_size
        self.all_step_sizes = []

        x = x_0
        y = x_0.copy()
        t_momentum = 1.0

        self._initial_storage(x, storage and _should_store(0), self.func(x), 0.0)
        self.all_step_sizes.append(self.step_size)

        if alternative_stopping_criterion_initial_call is not None:
            stop = alternative_stopping_criterion_initial_call(x, self)
        else:
            stop = False

        self.t.append(0)
        t_current = time.time()
        t_init = t_current

        while self._k < max_iter and not stop:
            grad_y = self.grad(y)
            f_y = self.func(y)

            # backtracking: inflate L until the descent lemma is satisfied
            L = self.L
            for _ in range(self.max_backtrack):
                x_new = self.project(y - (1.0 / L) * grad_y)
                diff = x_new - y
                if self.func(x_new) <= f_y + grad_y @ diff + (L / 2.0) * (diff @ diff):
                    break
                L *= self.eta
            else:
                warnings.warn(
                    f"Backtracking did not converge in {self.max_backtrack} steps "
                    f"at iteration {self._k}. Consider a smaller initial step_size.",
                    RuntimeWarning,
                    stacklevel=2,
                )
            self.L = L

            # FISTA momentum update
            t_new = (1.0 + float(np.sqrt(1.0 + 4.0 * t_momentum**2))) / 2.0

            xp = cp if not isinstance(x_new, np.ndarray) else np
            delta_x = float(xp.linalg.norm(x_new - x))

            y = x_new + (t_momentum - 1.0) / t_new * (x_new - x)
            x = x_new
            t_momentum = t_new

            self._k += 1
            self.all_step_sizes.append(1.0 / L)

            self.storage(
                x,
                storage=storage and _should_store(self._k),
                f=self.func(x),
                delta_x=delta_x,
            )

            if alternative_stopping_criterion is not None:
                stop = alternative_stopping_criterion(x, self)
            else:
                stop = self._stopping_criterion(
                    del_objective_tol, del_objective_n, del_x_tol, del_x_n
                )

            self._additional_action(x)
            self.t.append(time.time() - t_init)
            self.t_it.append(time.time() - t_current)
            t_current = time.time()

        self._post_step(x)
        self.all_step_sizes = np.array(self.all_step_sizes)
        return x
