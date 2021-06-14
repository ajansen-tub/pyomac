import numpy as np
from scipy.spatial.distance import cdist
from scipy.cluster.hierarchy import linkage, fcluster
from pyomac.misc import mac_value, err_rel


def ssi_cov(data: np.ndarray, fs: float, ts: float = 5., eps_fn: float = 0.02, eps_zeta: float = 0.05,
            eps_mac: float = 0.01, f_min: float = 0.5, f_max: float = 20., n_order: int = 100,
            t: float = 0.025, min_count: int = 15, return_poles: bool = False):
    """
    ssi-cov algorithm according to E. Cheynet, 2018

    see: https://de.mathworks.com/matlabcentral/fileexchange/69030-operational-modal-analysis-with-automated-ssi-cov-algorithm

    Parameters
    ----------
    data : array_like
        Time series of measurement values
    fs : float
        Sampling frequency [Hz] of the `data` time series
    ts : float, optional
        time lag [s] for covariance calculation. Defaults to 5. s
    eps_fn : float, optional
        Threshold [%] for frequency stability calculations. Defaults to 0.02
    eps_zeta : float, optional
        Threshold [%] for damping stability calculations. Defaults to 0.05
    eps_mac : float, optional
        Threshold [-] for mode shape stability calculations. Defaults to 0.01
    f_min : float, optional
        Lowest considered frequency. Defaults to 0.5 Hz
    f_max : float, optional
        Highest considered frequency. Defaults to 20. Hz
    n_order : int, optional
        Highest considered mode order. Defaults to 100
    t : float, optional
        Similarity threshold [%] as cut-off for hierachical clustering. Defaults to 0.025
    min_count : int, optional
        Minimal cluster size. Defaults to 15
    return_poles :bool, optional
        If True, ndarray containing the poles are returned, e.g. fo plotting a stability diagram. Defaults to False


    Returns
    -------
    fn : ndarray
        Identified natural frequencies [Hz]
    zeta : ndarray
        Identified modal damping [-]
    phi : ndarray
        Identified mode shapes [-]
    poles_stab : ndarray, optional
        Stable poles for stability diagram
    order_stab : ndarray, optional
        Corresponding order of stable poles for stability diagram
    poles : ndarray, optional
        All identified poles for stability diagram
    order : ndarray, optional
        Corresponding order of all poles for stability diagram

    Last changed: ajansen, 2021-06-01
    """

    if data.shape[1] < data.shape[0]:
        data = data.T

    n_ch = data.shape[0]
    dt = 1 / fs

    # get impulse response function via NExt
    irf = next_method(data, dt, ts)

    # get hankel matrix
    u, s = block_hankel(irf)

    # ToDo: The following is a bit of a mess! Is the stability check even necessary? Or would it be a better solution to
    #  cluster all poles? - ajansen, 2021-06-10

    poles = np.empty((n_ch + 2, 0))
    order = np.array([], dtype=int)
    stab = np.array([], dtype=bool)

    # get poles
    fn0, zeta0, ms0 = modal_id(u, s, n_order, n_ch, dt)

    mask = (fn0 > f_min) & (fn0 < f_max)
    fn0 = fn0[mask]
    zeta0 = zeta0[mask]
    ms0 = ms0[:, mask]

    poles_temp = np.zeros((ms0.shape[0] + 2, ms0.shape[1]))

    poles_temp[0] = fn0
    poles_temp[1] = zeta0
    poles_temp[2:] = ms0

    order_temp = np.ones_like(fn0, dtype=int) * n_order
    stab_temp = np.zeros_like(fn0).astype(bool)

    poles = np.concatenate((poles, poles_temp), axis=1)
    order = np.concatenate((order, order_temp))
    stab = np.concatenate((stab, stab_temp))

    for n_pole in range(n_order - 1, 2, -2):
        fn1, zeta1, ms1 = modal_id(u, s, n_pole, n_ch, dt)

        mask = (fn1 > f_min) & (fn1 < f_max)
        fn1 = fn1[mask]
        zeta1 = zeta1[mask]
        ms1 = ms1[:, mask]

        poles_temp = np.zeros((ms1.shape[0] + 2, ms1.shape[1]))

        poles_temp[0] = fn1
        poles_temp[1] = zeta1
        poles_temp[2:] = ms1

        order_temp = np.ones_like(fn1) * n_pole
        stab_temp = stability_check(fn0, zeta0, ms0, fn1, zeta1, ms1, eps_fn, eps_zeta, eps_mac)

        poles = np.concatenate((poles, poles_temp), axis=1)
        order = np.concatenate((order, order_temp))
        stab = np.concatenate((stab, stab_temp))

        fn0 = fn1
        zeta0 = zeta1
        ms0 = ms1

    poles_stab: np.ndarray = poles[:, stab]
    order_stab: np.ndarray = order[stab]

    # remove negative damping
    mask = poles_stab[1] > 0.
    poles_stab = poles_stab[:, mask]
    order_stab = order_stab[mask]

    # cluster poles
    res = get_cluster(poles_stab, t, min_count)

    fn = res[0]
    zeta = res[1]
    ms = res[2:, :].T

    # scale mode shape with absolute maximum
    ms = ms / np.abs(ms).max(axis=1).reshape(-1, 1)

    if return_poles:
        return fn, ms, zeta, poles_stab, order_stab, poles, order

    else:
        return fn, ms, zeta


def next_method(data, dt, ts):
    # FFT-based
    n_ch = data.shape[0]
    m = round(ts / dt)

    irf = np.zeros((n_ch, n_ch, m), dtype=np.complex)  # +1 ?

    for i in range(n_ch):
        for j in range(n_ch):
            y1 = np.fft.fft(data[i])
            y2 = np.fft.fft(data[j])

            h = np.fft.ifft(y1 * np.conj(y2))
            irf[i, j, :] = h[:m]

    return irf


def block_hankel(irf):
    # Toeplitz matrix

    n_len = round(irf.shape[2] / 2) - 1
    n_ch = irf.shape[0]
    toep = np.zeros((n_ch * n_len, n_ch * n_len), dtype=np.complex)

    for i in range(n_len):
        for j in range(n_len):
            toep[i * n_ch: (i + 1) * n_ch, j * n_ch: (j + 1) * n_ch] = irf[:, :, n_len + i - j]

    u, s, _ = np.linalg.svd(toep)

    return u, s


def modal_id(u, s, n_poles, n_ch, dt):
    # observability matrix

    obs_m = np.dot(u[:, : n_poles], np.diag(np.sqrt(s[:n_poles])))

    idx_obs = np.min([n_ch, obs_m.shape[0]])
    c = obs_m[:idx_obs, :]
    jb = int(round(obs_m.shape[0] / idx_obs))

    a = np.dot(np.linalg.pinv(obs_m[:idx_obs * (jb - 1), :]), obs_m[-idx_obs * (jb - 1):, :])
    w, v = np.linalg.eig(a)

    mu = np.log(w) / dt
    f_n = np.abs(mu[1::2]) / (2 * np.pi)
    zeta = -np.real(mu[1::2]) / np.abs(mu[1::2])
    phi = np.real(np.dot(c, v))
    phi = phi[:, 1::2]

    return f_n, zeta, phi


def stability_check(fn0, zeta0, phi0, fn1, zeta1, phi1, eps_fn, eps_zeta, eps_mac):
    y_fn = cdist(np.expand_dims(fn0, axis=1), np.expand_dims(fn1, axis=1), err_rel)
    y_zeta = cdist(np.expand_dims(zeta0, axis=1), np.expand_dims(zeta1, axis=1), err_rel)
    y_mac = cdist(phi0.T, phi1.T, mac_value)

    mask = (y_fn < eps_fn) & (y_zeta < eps_zeta) & (1 - y_mac < eps_mac)

    stability = mask.any(axis=0)

    return stability


def dist_cluster(x0, x1):
    return err_rel(x0[0], x1[0]) + (1 - mac_value(x0[2:], x1[2:]))


def get_cluster(poles, t, min_count):
    z = linkage(poles.T, method='single', metric=dist_cluster)
    clusters = fcluster(z, t=t, criterion='distance')

    unique, counts = np.unique(clusters, return_counts=True)
    valid_clusters = unique[counts > min_count]

    res = np.zeros((poles.shape[0], valid_clusters.shape[0]))

    for idx, n in enumerate(valid_clusters):
        idx_c = clusters == n

        res[0, idx] = np.median(poles[0, idx_c])
        res[1, idx] = np.median(poles[1, idx_c])
        res[2:, idx] = np.median(poles[2:, idx_c], axis=1)

    idx_sort = res[0].argsort()
    res = res[:, idx_sort]

    return res

