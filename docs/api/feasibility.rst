=========================
suppy.feasibility
=========================

Linear algorithms
=========================


Hyperslab AMS algorithms :math:`(lb \leq Ax \leq ub)`
----------------------------------------------------

.. autoclass:: suppy.feasibility.SequentialAMSHyperslab
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.SimultaneousAMSHyperslab
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.StringAveragedAMSHyperslab
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.BlockIterativeAMSHyperslab
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

Hyperplane AMS algorithms :math:`( Ax \leq b)`
--------------------------------------------------

.. autoclass:: suppy.feasibility.SequentialAMSHyperplane
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.SequentialAMSHyperplane
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.SimultaneousAMSHyperplane
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.StringAveragedAMSHyperplane
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.BlockIterativeAMSHyperplane
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
Halfspace AMS algorithms :math:`( Ax = b)`
--------------------------------------------------
.. autoclass:: suppy.feasibility.SequentialAMSHalfspace
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.SequentialAMSHalfspace
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.SimultaneousAMSHalfspace
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.StringAveragedAMSHalfspace
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.BlockIterativeAMSHalfspace
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:


ARM algorithms
-------------------------

.. autoclass:: suppy.feasibility.SequentialARM
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.SimultaneousARM
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility.StringAveragedARM
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:


ART3+ algorithms
-------------------------

.. autoclass:: suppy.feasibility.SequentialART3plus
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. .. autoclass:: suppy.feasibility.SimultaneousART3plus
..    :members:
..    :inherited-members:
..    :undoc-members:
..    :show-inheritance:


Split algorithms
=========================
Split feasibility problems have the goal of finding :math:`x \in C` such that :math:`Ax \in Q`. :math:`C` is a convex subset of the input space :math:`\mathscr{H}_1` and :math:`Q` a convex subset in the target space :math:`\mathscr{H}_2` with the two spaces connected by the linear operator :math:`A:\mathscr{H}_1 \rightarrow \mathscr{H}_2`.
The base class for split feasibility problems is :class:`SplitFeasibility`.

.. autoclass:: suppy.feasibility._split_algorithms.SplitFeasibility
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:

.. autoclass:: suppy.feasibility._split_algorithms.CQAlgorithm
   :members:
   :inherited-members:
   :undoc-members:
   :show-inheritance:
