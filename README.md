# suppy - Python superiorization library

`suppy` is a native python library for quick setup and evaluation of superiorzation algorithms.
The superiorization method uses underlying feasibility seeking algorithms, i.e. algorithms that have the goal of finding a point in the intersection of convex sets(
$
\text{find } \mathbf{x}\in C = \bigcap_i C_i
$),
and perturbs them with respect to an object function. Comparing it to constrained optimization, the goal is not to find an $\mathbf{x}$ that fully minimizes all the objectives, but rather find an $\mathbf{x}$ with reduced objective function.

For a mathematical introduction to the topic check [this paper](https://pmc.ncbi.nlm.nih.gov/articles/PMC2897099/). Furthermore a library citing general sources can be found [here](https://arxiv.org/pdf/1506.04219).


## Installation
Install simply via pip:

```
pip install suppy
```

In general suppy works on the CPU (`numpy`), as well as the GPU(`cupy`). The standard installation will not set up `cupy`.
For installation check your CUDA version and the [cupy installation guide](https://docs.cupy.dev/en/stable/install.html).

## Usage

`suppy` is a modular library for building and running superiorization models.
it allows for quick setup of feasibility seeking and superiorization algorithms.

Projections for simple structures like halfspaces or circles have dedicated implementations in the `suppy.projections` submodule and allow for visualization of the structure if they are 2D objects.

```
import numpy as np
from suppy.projections import HalfspaceProjection
a = np.array([1.,1.])
b = 3
halfspace = HalfspaceProjection(a,b)
halfspace.visualize()
```

To calculate the projection each ``Projection`` object has a dedicated function (which if possible changes the array in place):
```
x = np.array([4.,4.])
x_p = halfspace.project(x)
```

### Combined projections
For more complex structures that are formulated as the intersection of multiple constraints implementations for standard methods (``SequentialProjection``,``SimultaneousProjection``) and combinations of the two (``StringAveragedProjection``,``BlockIterativeProjection``) are available.

```
import numpy as np
from suppy.projections import BallProjection,SequentialProjection
center_1 = np.array([1,1])
radius_1 = 2
center_2 = np.array([-0.5,-1])
radius_2 = 1
ball_1 = BallProjection(center_1,radius_1)
ball_2 = BallProjection(center_2,radius_2)
joined_projection = SequentialProjection([ball_1,ball_2])
```
For a single projection step of these methods the ``project`` function can be used again. However since one step often times is not enough to find a point in the intersection, an entire run is needed.
This can be done by calling the ``solve`` method.

```
joined_projection.solve(np.array([3,3]))
```

### Linear algorithms

For linear problems $Ax\leq b$ dedicated implementations are available (right now formulated as $lb \leq Ax \leq ub$). In general ``AMS``, ``ARM``(Automatic relaxation method) and a sequential ``ART3+`` implementation are available.

```
import numpy as np
from suppy.feasibility import SequentialAMS
A = np.array([[1,1],[1,0]])
lb = np.array([-2,-2])
ub = np.array([2,2])
alg = SequentialAMS(A,lb,ub)
alg.solve(np.array([5.,5.]),max_iter = 100)
```

In general these algorithms have the same functions as the basic projections (``solve()``,``project()``,etc.) and can be used as part of joined projections.

## Superiorization

For setting up superiorzation models a feasibility seeking algorithm as well as a perturbation scheme are required.
For gradient step based perturbations this can be set up in the following way:
```
from suppy.perturbations import PowerSeriesGradientPerturbation

def objective(x):
    return 1/len(x)*(x**2).sum(axis = 0)

def gradient(x):
    grad = 1/len(x)*2*x
    return grad/np.sqrt(np.sum(grad**2))

pert = PowerSeriesGradientPerturbation(objective,gradient)

from suppy.superiorization import Superiorization

sup_model = Superiorization(alg,pert)
sup_model.solve(np.array([3,2]), 1000)
```


## Funding and Development
`suppy` was developed as part of the German-Israeli Cooperation in Cancer Research (Project Ca 216)
Responsible for suppy are:

- Tobias Becher (DKFZ, Lead Developer & Research Fellow)
- Prof. Yair Censor (University of Haifa, Principal Investigator / Scientific Lead)
- Dr. Niklas Wahl (DKFZ, Principal Investigator / Scientific Lead)
<p align="right">
<img src= https://raw.githubusercontent.com/e0404/matRad/98ba2fb8b07f727a3963cf2572c82a548444580b/matRad/gfx/dkfz_logo_blue.png width="200" />
</p>
