.. _user_guide:

User guide
==========

*SupPy* is a library for projection based feasibility seeking and superiorization algorithms.

Feasibility seeking algorithms have the goal of finding a point in a constrained set given by the intersection of individual constraints :math:`C_i`

.. math::
    \begin{equation}
    \text{find } x \in C=\bigcap_{i}  C_i
    \end{equation}


..    \begin{aligned}
..    & \underset{x}{\text{minimize}}
..    & & f(x) \\
..    & \text{subject to}
..    & & g_i(x) \leq 0, \quad i = 1, \dots, m \\
..    & & & h_j(x) = 0, \quad j = 1, \dots, p
..    \end{aligned}

One way to achieve this is by means of projecting onto the individual sets :math:`P_{C_i}(x)`.

Individual projections
----------------------

Individual constraints in *SupPy* are represented by :class:`Projection` objects that each have a method :meth:`project` that calculates the projection of a given point `x` onto the set.
Furthermore a relaxed projection can be calculated by setting up a relaxation parameter when constructing the object. :meth:`project` then calculates a relaxed projection based on the parameter. For indication on whether to use the GPU for calculations a `use_gpu` parameter is used. This is either set automatically if the required input allows to determine it, or is set manually when constructing the object.


For simple objects like hyperplanes or balls analytical formulations based on the metric projection can be found.
A list and explanation of these simple projections can be found in the :ref:`projections` section.


Should the constraint be formulated as a level set of a function (i.e. :math:`C_i = \{x \in \mathbb{R}^n | f_i(x) \leq \alpha_i\}`) the projection can be calculated by means of the subgradient projection method:

.. math::

   P_{C_i}(x) =
   \begin{cases}
   x - \frac{(f_i(x)-\alpha_i)_+}{||g_i(x)||^2} g_i(x) & \text{if } g_i(x) \neq 0 \\
   x & \text{otherwise}
   \end{cases}

How these constraints can be implemented is found in the :ref:`subgradient_projections` section.


Projection onto the intersection of sets
----------------------------------------

To project onto the intersection of general sets, combinations of the individual projections are used.
The simplest ideas are :class:`SequentialProjection` and :class:`SimultaneousProjection` methods that project onto the individual sets sequentially or simultaenously, respectively.
Furthermore combinations of the two approaches can be used in the form of :class:`StringAveragedProjection` and :class:`BlockIterativeProjection`.
For a full readup on how these are implemented in *SupPy* please refer to the :ref:`projection_methods` section.



Linear feasibility problems
---------------------------
While individual linear constraints :math:`C_i = \{x \in \mathbb{R}^n | a_i x \leq b_i\}` can be projected onto using the dedicated :class:`HalfspaceProjection` class, this becomes cumbersome for the intersection of many linear constraints.
To speed up the computation and setup for projections onto linear constraints, *SupPy* provides several matrix based formulations in the :mod:`feasibility` module.
These include AMS, ARM and ART3+ algorithms. For a full readup on these algorithms please refer to the :ref:`feasible` section.


Superiorization
===============
The idea of superiorization can be best explained by comparing it to constrained optimization.
A general constrained optimization problem can be formulated as:

.. math::

   \begin{aligned}
   & \underset{x}{\text{minimize}}
   & & f(x) \\
   & \text{subject to}
   & & x \in C=\bigcap_{i}  C_i
   \end{aligned}

The goal of constrained optimization is to fully minimize the objective function while meeting all of the constraints.
Superiorization meanwhile aims to find a point in the feasible set and only reducing the objective function value - not necessarily minimizing it.
This is done by using a feasibility seeking algorithm and perturbing it with respect to the objective function :math:`f` to reduce its value.

In *SupPy* superiorization algorithms are class based and can be found in the :mod:`superiorization` module.
For set up an underlying feasibility seeking algorithm as well as perturbation scheme is needed.
An explanation on how this is done can be found in the :ref:`superiorization` section.

.. toctree::
   :maxdepth: 1
   :caption: User guide:

   projections
   projection_methods
   feasibility
   superiorization
