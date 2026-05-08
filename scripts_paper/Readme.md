# Set up of scripts used in the paper
This file gives a quick introduction on how to reproduce the "Modular and
GPU-accelerated superiorization with SupPy" paper were obtained".

## Seismic Tomography
The notebook in the seismic tomography folder was used to run the calculation in SupPy.
The data required was generated using the *seismic_data_generation.m* script, which was also used for timing of the DROP method. (Running requires installation/functionality of the [AIR Tools II library](https://github.com/jakobsj/AIRToolsII)). - Was run with Matlab R2023b.

## CT reconstruction
The CT slice data was taken from the [LoDoPaB-CT](https://www.nature.com/articles/s41597-021-00893-z) dataset In particular slice 14 from *observation_test_000.hdf5*.
The reconstruction matrix is not directly available but can be extracted from the astra toolbox.
In this case the *get_impl* file was used which is a condensed version of [this file](https://github.com/jleuschn/lodopab_tech_ref/blob/master/resimulate_observations.py) in the LoDoPaB-CT technical documentation.
For extraction either a small code snippet has to be inserted after line 146 in [this file](https://github.com/odlgroup/odl/blob/release-candidate/0.8.3/odl/tomo/backends/astra_cpu.py) of the odl library (v0.8.3), or a breakpoint can used.

Code snippet to insert:

    matrix_id = astra.projector.matrix(proj_id)
    S = astra.matrix.get(matrix_id)
    import scipy
    scipy.sparse.save_npz('CT',S)


## Radiotherapy treatment planning
The calculation of geometry and dose-influence matrix requires the [matRad](https://github.com/e0404/matRad) treatment planning system. For TG119 and the head-and-neck patient the *TG119.m* and *HN.m* scripts can be used, respectively. The optimization reference shown for the head-and-neck patient can also be calculated by commenting out the appropriate section.
