from itertools import cycle
import numpy as np

import numpy.typing as npt
import matplotlib.pyplot as plt


def plot2d_linear_constraints(x: npt.NDArray, A: npt.NDArray, lb: npt.NDArray, ub: npt.NDArray):
    """
    Plots the linear 2d constraints defined by lb <= A_0*x+A_1*y <= ub.

    Parameters:
    x (array-like): The x-values for the plot.
    A (array-like): The coefficient matrix of the linear constraints.
    lb (array-like): The lower bounds of the constraints.
    ub (array-like): The upper bounds of the constraints.

    Returns:
    None
    """
    plt.figure()
    ax = plt.gca()

    color_cycle = cycle(plt.cm.tab10.colors)
    for i in range(A.shape[0]):
        color = next(color_cycle)
        if A[i, 1] == 0:
            ax.axvline(x=-lb[i] / A[i, 0], label=f"Constraint {i}", color=color)
            ax.axvline(x=-ub[i] / A[i, 0], label=f"Constraint {i}", color=color)
        else:
            y = (lb[i] - A[i, 0] * x) / A[i, 1]
            ax.plot(x, y, label=f"Constraint {i}", color=color)
            y = (ub[i] - A[i, 0] * x) / A[i, 1]
            ax.plot(x, y, label=f"Constraint {i}", color=color)
    ax.set_ylim(-2, 2)
    ax.legend()
    return plt.gca()


def get_linear_constraint_bounds(x: npt.NDArray, A: npt.NDArray, lb: npt.NDArray, ub: npt.NDArray):
    """
    For a given 1D array x and linear constraints defined by lb <=
    A_0*x+A_1*y <= ub, this function returns the associated y values.

    Parameters:
    x (npt.NDArray): Array-like object representing the x values.
    A (npt.NDArray): Array-like object representing the coefficients of the linear constraints.
    lb (npt.NDArray): Array-like object representing the lower bounds of the constraints.
    ub (npt.NDArray): Array-like object representing the upper bounds of the constraints.

    Returns:
    x_all (npt.NDArray): Array-like object with size (len(x), len(lb)*2) storing all x values for all constraints (lower and upper bounds)
    y_all (npt.NDArray): Array-like object storing all associated y values
    """
    y_l = (lb - A[:, 0] * x[:, None]) / A[
        :, 1
    ]  # gives a nxm matrix with n = len(x) and m = len(lb)
    y_u = (ub - A[:, 0] * x[:, None]) / A[:, 1]

    x_all = np.tile(
        x, (A.shape[0] * 2, 1)
    ).T  # get a x matrix that stores all x values for all constraints
    return x_all, np.concatenate((y_l, y_u), axis=1)


def plot3d_linconstrained_function(func, A, x, lb, ub, func_args=()):
    """
    For given x values finds the associated y values (giving the linear
    constrained bounds)
    and plots the 2D surface of a function with linear constraints.

    Parameters:
    A (npt.NDArray): Array-like object representing the coefficients of the linear constraints.
    x (npt.NDArray): Array-like object representing the x values.
    lb (npt.NDArray): Array-like object representing the lower bounds of the constraints.
    ub (npt.NDArray): Array-like object representing the upper bounds
    """
    x_1, x_2 = get_linear_constraint_bounds(x, A, lb, ub)

    x_reshaped = np.reshape(
        np.array([x_1, x_2]), (2, x_1.shape[0] * x_1.shape[1])
    )  # reshape the array to be able to multiply with A
    y = func(x_reshaped, *func_args).reshape(x_1.shape[0], x_1.shape[1])

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for i in range(x_1.shape[1]):
        ax.plot(x_1[:, i], x_2[:, i], y[:, i], color="xkcd:black")

    x_grid = np.linspace(np.min(x_1), np.max(x_1), 100)
    y_grid = np.linspace(np.min(x_2), np.max(x_2), 100)
    X, Y = np.meshgrid(x_grid, y_grid)
    Z = func(np.array([X, Y]).reshape(2, -1)).reshape(X.shape[0], X.shape[1])
    ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.2)
    ax.set_xlabel("x$_1$")
    ax.set_ylabel("x$_2$")
    ax.set_xlim(np.min([np.min(x_1), np.min(x_2)]), np.max([np.max(x_1), np.max(x_2)]))
    ax.set_ylim(np.min([np.min(x_1), np.min(x_2)]), np.max([np.max(x_1), np.max(x_2)]))
    return fig, ax


def plot3d_general_objects(func, objects, func_args=(), x: None | npt.NDArray = None):
    """PLots a 3D function with multiple objects."""
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    for el in objects:
        xy = el.get_xy()  # should return a 2xn array
        ax.plot(xy[0, :], xy[1, :], func(xy, *func_args), color="xkcd:black")

    if x is None:
        x = np.linspace(-10, 10, 100)

    X, Y = np.meshgrid(x, x)
    Z = func(np.array([X, Y]).reshape(2, -1), *func_args).reshape(X.shape[0], X.shape[1])
    ax.plot_surface(X, Y, Z, cmap="plasma", alpha=0.2)

    return fig, ax


# if __name__ == "__main__":
#     #Importing some libraries
#     import numpy as np
#     import matplotlib.pyplot as plt
#     import suppy.projections as pr
#     import suppy.superiorization.sup as sup

#     center_1 = np.array([0,0])
#     radius = 1
#     center_2 = np.array([1,1])

#     center_3 = np.array([1.5,0])

#     #Creating a circle

#     Ball_1 = pr.Ball_Projection(center_1, radius)
#     Ball_2 = pr.Ball_Projection(center_2, radius)

#     def func_1(x):
#         return 1/len(x)*(x**2).sum(axis=0)

#     def grad_1(x):
#         return 1/len(x)*2*x

#     Proj = pr.Sequential_Projection([Ball_1,Ball_2])

#     Test = sup.Standard_superiorize(Proj,func_1,grad_1)

#     xF = Test.solve(np.array([1,3]),1,10,storage=True)
#     X = np.array(Test.all_X_values)

#     fig,ax = plot3d_generalObjects(func_1,[Ball_1,Ball_2],x=np.linspace(-3,3,50))
#     ax.plot(X[:,0],X[:,1],Test.all_function_values)
#     plt.show()
