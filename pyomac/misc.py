import numpy as np


def mac_value(ms0, ms1):
    mac = np.abs(ms0.T @ ms1) ** 2 / (ms0.T @ ms0) / (ms1.T @ ms1)

    if mac > 1.0:
        mac = 1.0

    return mac


def err_rel(x0, x1):
    err = np.abs((x1 - x0) / x0)

    return err
