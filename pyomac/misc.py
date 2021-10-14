import numpy as np


def mac_value(ms0, ms1):
    mac = np.abs(ms0.T @ ms1) ** 2 / (ms0.T @ ms0) / (ms1.T @ ms1)

    if mac > 1.0:
        mac = 1.0

    return mac


def err_rel(x0, x1):
    err = np.abs((x1 - x0) / x0)

    return err


def MPD(phi: np.ndarray) -> float:
    """Calculate the Mean Phase Deviation (MPD) of a modeshape vector.

    Based on article:
    Reynders, E., Houbrechts, J., & De Roeck, G. (2012). Fully automated (operational) modal analysis. Mechanical Systems and Signal Processing, 29, 228-250.

    Parameters
    ----------
    phi : np.ndarray
        1-dimensional modeshape vector

    Returns
    -------
    float
        Mean Phase Deviation of the modeshape vector
    """    """"""

    # 1. recover the mean phase
    ms_matrix = np.array([np.real(phi), np.imag(phi)]).T
    _, _, V = np.linalg.svd(ms_matrix, full_matrices=False)
    # Use nomenclature according to the cited source
    V_12 = V[0, 1]
    V_22 = V[1, 1]
    # weighting factors:
    w = np.abs(phi)
    # alternative:
    # w = np.ones(phi.shape)
    MPD = np.sum(w * np.arccos(np.abs(np.real(phi)*V_22 - np.imag(phi)
                 * V_12) / (np.sqrt(V_12**2 + V_22**2) * np.abs(phi)))) / np.sum(w)
    return MPD
