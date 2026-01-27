=========================
suppy.feasibility
=========================

Linear algorithms
=========================


Hyperslab algorithms :math:`(lb \leq Ax \leq ub)`
------------------------------------------------------

.. currentmodule:: suppy.feasibility

.. autosummary::
   :toctree: generated/Hyperslabs/

   SequentialAMSHyperslab
   SimultaneousAMSHyperslab
   StringAveragedAMSHyperslab
   BlockIterativeAMSHyperslab


Hyperplane algorithms :math:`( Ax \leq b)`
--------------------------------------------------
.. autosummary::
   :toctree: generated/Hyperplane/

   KaczmarzMethod
   SimultaneousKaczmarzMethod
   StringAveragedKaczmarz
   BlockIterativeKaczmarz


Halfspace algorithms :math:`( Ax = b)`
--------------------------------------------------

.. autosummary::
   :toctree: generated/Halfspaces/

   SequentialAMSHalfspace
   SimultaneousAMSHalfspace
   StringAveragedAMSHalfspace
   BlockIterativeAMSHalfspace

ARM algorithms
-------------------------

.. autosummary::
   :toctree: generated/ARMs/

   SequentialARM
   SimultaneousARM
   StringAveragedARM


.. ART3+ algorithms
.. -------------------------

.. .. autosummary::
..    :toctree: generated/ART3/

..    SequentialART3
..    SimultaneousART3


Split feasibility
=========================
Split feasibility problems have the goal of finding :math:`x \in C` such that :math:`Ax \in Q`. :math:`C` is a convex subset of the input space :math:`\mathscr{H}_1` and :math:`Q` a convex subset in the target space :math:`\mathscr{H}_2` with the two spaces connected by the linear operator :math:`A:\mathscr{H}_1 \rightarrow \mathscr{H}_2`.
The base class for split feasibility problems is :class:`SplitFeasibility`.

Split algorithms
-------------------------
.. autosummary::
   :toctree: generated/Split/

   CQAlgorithm

Underlying base class
--------------------------------

.. currentmodule:: suppy.feasibility._split_algorithms

.. autosummary::
   :toctree: generated/Split_base/

   SplitFeasibility
