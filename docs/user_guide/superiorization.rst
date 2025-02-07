.. _superiorization:

Superiorization
================
So far, models focused only on feasibility seeking with the goal of finding a point in the intersection of convex sets :math:`C_i` :math:`\text{find } x \in C=\bigcap_{i}  C_i`.
Superiorization on the other hand not only aims to find a point in the feasible, set but also has the goal of reducing, not necesarily minimizing an additional objective function :math:`f(x)`.
This is done by using a feasibility seeking algorithm and perturbing it with respect to the objective function :math:`f` to reduce its value.

In *SupPy* superiorization algorithms are class based and can be found in the :mod:`suppy.superiorization` module.
For set up an underlying feasibility seeking algorithm as well as perturbation scheme is needed.

Perturbation schemes
=====================

While perturbations can be performed in any way, as long as they reduce the objective function value, right now only gradient based perturbations are implemented.
The following scheme is available right now:

* :class:`~suppy._perturbations._base.PowerSeriesGradientPerturbation`

Gradient steps are performed with a step size :math:`\alpha` that decreases according to a power law :math:`\alpha^l` with :math:`0 < \alpha < 1` and :math:`l` increasing after each computation.
Set up of a perturbation scheme is done in the following way:

.. code-block:: python
    import numpy as np
    from suppy.perturbations import PowerSeriesGradientPerturbation
    PowerSeriesGradientPerturbation

    #define objective function and gradient
    def func_1(x):
    return 1/len(x)*(x**2).sum(axis = 0)

    def grad_1(x):
        grad = 1/len(x)*2*x
        return grad/np.sqrt(np.sum(grad**2))

    pert = PowerSeriesGradientPerturbation(func_1,grad_1)


For a superiorization scheme we need the underlying feasibility seeking algorithm and the perturbation scheme.
.. code-block:: python
    from suppy.projections import BandProjection, SequentialProjection
    band_1 = BandProjection(np.array([1,2]), 3,4)
    band_2 = BandProjection(np.array([-1,1.5]), 1,2)
    seq_proj = SequentialProjection([band_1, band_2])

With those two we can set up a superiorization scheme and solve it:
.. code-block:: python
    # import the superiorization package
    from suppy.superiorization import Superiorization
    sup_model = Superiorization(seq_proj,pert)
    sup_model.solve(np.array([3,2]), 1000,storage = True)
