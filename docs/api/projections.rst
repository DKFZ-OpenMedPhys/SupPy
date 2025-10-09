.. _projections_api:

suppy.projections
=========================

This module implements a framework for general projection methods.

Base class for all projections
---------------------------------

.. currentmodule:: suppy.projections._projections

.. autosummary::
   :toctree: generated/PrivateProjections/

   Projection


BasicProjections
-------------------


.. currentmodule:: suppy.projections

.. autosummary::
   :toctree: generated/BasicProjections/

   BoxProjection
   WeightedBoxProjection
   BallProjection
   HalfspaceProjection
   BandProjection
   MaxDVHProjection
   MinDVHProjection

.. currentmodule:: suppy.projections._projections

Underlying base class for basic projections
------------------------------------------------
.. autosummary::
   :toctree: generated/PrivateProjections/

   BasicProjection



Projection methods
----------------------

Methods to project onto the intersection of a constraint set.

Public projection methods
-----------------------------
.. currentmodule:: suppy.projections

.. autosummary::
   :toctree: generated/projection_methods/

   SequentialProjection
   SimultaneousProjection
   StringAveragedProjection
   BlockIterativeProjection

Underlying base class for projection methods
------------------------------------------------

.. currentmodule:: suppy.projections._projection_methods

.. autosummary::
   :toctree: generated/base_projection_methods/

   ProjectionMethod
