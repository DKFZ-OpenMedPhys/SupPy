.. _projection_methods:

Projection methods
==================

Projection methods are the second part of the :class:`~suppy.projections._projections.Projection` module.
They allow to find the intersection of convex sets, by projecting onto them sequentially, simultaneously, or in a combined fashion.
They are implented as subclasses of the :class:`~suppy.projections._projection_methods.ProjectionMethod` class with the individual projections passed as a list to the constructor.

SequentialProjection
--------------------

The ``SequentialProjection`` class allows to project onto multiple sets sequentially.

.. code-block:: python

    import numpy as np
    from suppy.projections import BoxProjection
    from suppy.projections import SequentialProjection
    a_1 = np.array([0, 0])
    b_1 = np.array([1, 1])
    a_2 = np.array([0, 0])
    b_2 = np.array([1, 2])
    projection_1 = BoxProjection(a_1, b_1)
    projection_2 = BoxProjection(a_2, b_2)
    sequential_projection = SequentialProjection([projection_1, projection_2])
    x = np.array([2, 2])
    sequential_projection.project(x)

where ``projection_1`` and ``projection_2`` are two individual box projections.
The order of the projections by default is the same as the order in which the projections are passed in to the constructor. It can be changed by passing a control sequence (``cs``) as an optional input argument to the constructor.


SimultaneousProjection
----------------------

The ``SimultaneousProjection`` class allows to project onto multiple sets simultaneously.

.. code-block:: python

    import numpy as np
    from suppy.projections import BoxProjection
    from suppy.projections import SimultaneousProjection
    a_1 = np.array([0, 0])
    b_1 = np.array([1, 1])
    a_2 = np.array([0, 0])
    b_2 = np.array([1, 2])
    projection_1 = BoxProjection(a_1, b_1)
    projection_2 = BoxProjection(a_2, b_2)
    weights = np.array([1/3, 2/3])
    simultaneous_projection = SimultaneousProjection([projection_1, projection_2],weights)
    x = np.array([2, 2])
    simultaneous_projection.project(x)

where ``projection_1`` and ``projection_2`` are two individual box projections and ``weights`` are the weights assigned to the individual projections.
Mathematically the simultaneous projection is calculated by:

.. math::

    P(\mathbf{x}) = \sum_{i=1}^{n} w_i P_i(\mathbf{x})

where :math:`P_i` are the individual projections and :math:`w_i` are the respective weights.
If no weights :math:`w_i` are passed to the constructor they are chosen to be equal and sum up to 1. Should the passed weights not sum up to 1, a normalization is applied.
As of now, while the projections are performed simultaneously mathematically, the code performs them sequentially.

BlockIterativeProjection
------------------------

The ``BlockIterativeProjection`` class allows to split the constraints into different blocks, that are projected onto sequentially. The individual blocks themselves are projected onto using a simultaneous projection.
