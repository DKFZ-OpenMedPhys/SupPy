.. _feasible:


Feasibility seeking algorithms
==============================

While individual linear constraints :math:`C_i = \{x \in \mathbb{R}^n | a_i x \leq b_i\}` can be projected onto using the dedicated :class:`HalfspaceProjection` class, this becomes cumbersome for the intersection of many linear constraints.
For linear feasibility problems *SupPy* provides several implementations of feasibility seeking algorithms that take advante of matrix operations in the

These cover the following problems:

* Hyperplane :math:`Ax = b`
* Halfspace :math:`Ax \leq b`
* Hyperslab/Bands :math:`lb \leq Ax \leq ub`

For all three implementations the AMS algorithm/Kamzarz's method are available (sequentially and simultaenously).
Furthermore the ART3+ (Algebraic reconstruction technique) and ARM (Automatic relaxation method) algorithms are available for the hyperslab/bands problems.
Once a class is initialized a single step of the algorithm can be performed using the :meth:`project()` method, while a full run can be done using the :meth:`solve()` method.

Initialization of Halfspace and Hyperplane problems are identical, while the hyperslab/bands problem requires a lower and an upper bound (instead of a single bound).
A simple sequential model for :math:`lb \leq Ax \leq ub` can bet set up and solved using:

.. code-block:: python

    import numpy as np
    from suppy.feasibility import SequentialAMSHyperslab

    A = np.array([[1, 1], [-1, 1], [1, 0], [0, 1]])
    lb = np.array([-2, -2, -3 / 2, -3 / 2])
    ub = -1 * lb
    seq_model = SequentialAMSHyperslab(A, lb, ub)
    x_0 = np.array([10,2])
    x_sol = seq_model.solve(x_0,3000)

In the following variables for the different classes are explained.



Sequential implementations
--------------------------
Sequential implementations can be passed a control sequence `cs` that determines the order in which the constraints are processed. If no sequence is passed the constraints are evaluated in order of the rows.

Simultaneous implementations
----------------------------
Simultaneous implementations can be passed a `weights` option that determines how much an individual constraint :math:`\langle A,x \rangle \leq b` is weighted in the individual projection step. To ensure that `weights` add up to 1, the passed weights are divided by their sum.
If no `weights` are passed uniform weights are applied.


Hyperslab/Bands
================

* :class:`~suppy.feasibility._bands._ams_algorithms.SequentialAMSHyperslab`
* :class:`~suppy.feasibility._bands._ams_algorithms.SimultaneousAMSHyperslab`
* :class:`~suppy.feasibility._bands._ams_algorithms.BlockIterativeAMSHyperslab`
* :class:`~suppy.feasibility._bands._arm_algorithms.StringAveragedAMSHyperslab`
* :class:`~suppy.feasibility._bands._arm_algorithms.SequentialARMHyperslab`
* :class:`~suppy.feasibility._bands._arm_algorithms.SimultaneousARMHyperslab`
* :class:`~suppy.feasibility._bands._art3p_algorithms.SequentialART3PHyperslab`

Halfspace
=========
* :class:`~suppy.feasibility._halfspace._ams_algorithms.SequentialAMSHalfspace`
* :class:`~suppy.feasibility._halfspace._ams_algorithms.SimultaneousAMSHalfspace`
* :class:`~suppy.feasibility._halfspace._ams_algorithms.BlockIterativeAMSHalfspace`
* :class:`~suppy.feasibility._halfspace._arm_algorithms.StringAveragedAMSHalfspace`

Hyperplane
==========
* :class:`~suppy.feasibility._hyperplane._ams_algorithms.SequentialAMSHyperplane`
* :class:`~suppy.feasibility._hyperplane._ams_algorithms.SimultaneousAMSHyperplane`
* :class:`~suppy.feasibility._hyperplane._ams_algorithms.BlockIterativeAMSHyperplane`
* :class:`~suppy.feasibility._hyperplane._arm_algorithms.StringAveragedAMSHyperplane`
