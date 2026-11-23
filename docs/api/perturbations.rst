suppy.perturbations
===========================

The Perturbations module contains the underlying perturbation strategies that are usable in the superiorization algorithms.

Public perturbation strategies
-------------------------------

.. currentmodule:: suppy.perturbations

.. autosummary::
   :toctree: generated/PublicPerturbations/

   PowerSeriesGradientPerturbation

Experimental perturbation strategies
------------------------------------

These strategies are not part of the public API and may change without
notice. Import them from ``suppy.perturbations._variants``.

.. currentmodule:: suppy.perturbations._variants

.. autosummary::
   :toctree: generated/Privateperturbations/

   BarzilaiBorweinPerturbation

Underlying base classes
-----------------------------
.. currentmodule:: suppy.perturbations._base

.. autosummary::
   :toctree: generated/Privateperturbations/

   Perturbation
   ObjectivePerturbation
   GradientPerturbation
