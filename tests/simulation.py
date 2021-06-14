import numpy as np
from statespace_model import solver
from scipy.spatial.distance import cdist
from pyoma.misc import mac_value, err_rel


def generate_impulse_data(m, k, c_damp):
    # simulation parameters
    fs = 300
    time_step = 1 / fs
    t_end = 60

    t = np.arange(0, t_end, time_step)

    # force vector
    force = np.zeros((t.shape[0], m.shape[0]))
    force[:200, 2] = 200.

    t_impulse, acc_impulse = solver(m, k, c_damp, force, t)

    return t_impulse, acc_impulse, fs


def generate_random_data(m, k, c_damp):
    # simulation parameters
    fs = 300
    time_step = 1 / fs
    t_end = 5 * 60  # 5 min

    t = np.arange(0, t_end, time_step)

    # force vector
    np.random.seed(43)
    force = np.random.normal(size=(t.shape[0], m.shape[0]), ) * 100

    t_random, acc_random = solver(m, k, c_damp, force, t)

    return t_random, acc_random, fs


def assert_modal_identification(test_obj, test_title: str,
                                f_e: np.ndarray, ms_e: np.ndarray, zeta: np.ndarray = None,
                                threshold_f: float = 0.025, threshold_ms: float = 0.95, threshold_zeta: float = 0.1):
    print('---------')
    print(test_title)
    print('Identified natural frequencies: ' + str(f_e))
    print('Analytical (undamped) natural frequencies: ' + str(test_obj.f_a))

    # pairwise distance between estimated (e) and analytical (a) solution
    d_f = cdist(f_e.reshape(-1, 1), test_obj.f_a.reshape(-1, 1), err_rel).min(axis=1)
    d_ms = cdist(ms_e, test_obj.ms_a.T, mac_value).max(axis=1)

    if zeta is not None:
        d_zeta = cdist(zeta.reshape(-1, 1), test_obj.zeta_a.reshape(-1, 1), err_rel).min(axis=1)

    print('Rel. error natural frequencies: ' + str(d_f))
    print('MAC: ' + str(d_ms))

    if zeta is not None:
        print('Rel. error modal damping: ' + str(d_zeta))

    test_obj.assertTrue((d_f < threshold_f).all())
    test_obj.assertTrue((d_ms > threshold_ms).all())

    if zeta is not None:
        test_obj.assertTrue((d_zeta < threshold_zeta).all())
