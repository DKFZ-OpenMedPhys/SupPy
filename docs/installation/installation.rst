.. _installation:

Installation
============

Python distribution
-------------------

*SupPy* can be simply installed via pip:

.. code-block:: bash

    pip install suppy

Alternatively the source code can be found and cloned from `GitHub <https://github.com/DKFZ-OpenMedPhys/SupPy>`_.


If you would like to take advantage of *SupPy*'s GPU capabilities, the `cupy library <https://cupy.dev/>`_ is required.
For installation check your CUDA version and the `installation guide <https://docs.cupy.dev/en/stable/install.html>`_.

If your plan is to contribute to the development of *SupPy*, please fork the the `GitHub repository <https://github.com/DKFZ-OpenMedPhys/SupPy>`_ and install the package in editable mode with the dev option.

.. code-block:: bash

    pip install -e .[dev]
