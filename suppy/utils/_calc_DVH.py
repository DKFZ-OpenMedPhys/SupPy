import numpy as np
import matplotlib.pyplot as plt


def calc_DVH(dose, idxs: list):
    """"""

    d_min = np.min(dose)
    d_max = np.max(dose)
    n = 1000
    dose_grid = np.linspace(0, 1.05 * d_max, n)
    dvh = []
    for struct in idxs:
        dvh.append((dose[struct, None] > dose_grid).sum(axis=0) / (struct.sum()))
    return dvh, dose_grid
