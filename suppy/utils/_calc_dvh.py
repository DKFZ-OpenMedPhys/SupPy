import numpy as np


def calc_dvh(dose, idxs: list):
    """"""

    d_min = np.min(dose)
    d_max = np.max(dose)
    n = 1000
    dose_grid = np.linspace(0, 1.05 * d_max, n)
    dvh = []
    for struct in idxs:
        n_struct = dose[struct].shape[0]
        dvh.append((dose[struct, None] >= dose_grid).sum(axis=0) / n_struct)
    return dvh, dose_grid
