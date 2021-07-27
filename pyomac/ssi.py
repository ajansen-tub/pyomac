"""Stochastic Subspace Identification (SSI) module for pyomac."""
from typing import List, Tuple

import numpy as np
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.spatial.distance import cdist

from pyomac.misc import err_rel, mac_value


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
        stab_temp = stability_check(
            fn0, zeta0, ms0, fn1, zeta1, ms1, eps_fn, eps_zeta, eps_mac)

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
            toep[i * n_ch: (i + 1) * n_ch, j * n_ch: (j + 1)
                 * n_ch] = irf[:, :, n_len + i - j]

    u, s, _ = np.linalg.svd(toep)

    return u, s


def modal_id(u, s, n_poles, n_ch, dt):
    # observability matrix

    obs_m = np.dot(u[:, : n_poles], np.diag(np.sqrt(s[:n_poles])))

    idx_obs = np.min([n_ch, obs_m.shape[0]])
    c = obs_m[:idx_obs, :]
    jb = int(round(obs_m.shape[0] / idx_obs))

    a = np.dot(np.linalg.pinv(
        obs_m[:idx_obs * (jb - 1), :]), obs_m[-idx_obs * (jb - 1):, :])
    w, v = np.linalg.eig(a)

    mu = np.log(w) / dt
    f_n = np.abs(mu[1::2]) / (2 * np.pi)
    zeta = -np.real(mu[1::2]) / np.abs(mu[1::2])
    phi = np.real(np.dot(c, v))
    phi = phi[:, 1::2]

    return f_n, zeta, phi


def stability_check(
    fn0: np.ndarray,
    zeta0: np.ndarray,
    phi0: np.ndarray,
    fn1: np.ndarray,
    zeta1: np.ndarray,
    phi1: np.ndarray,
    eps_fn: float,
    eps_zeta: float,
    eps_mac: float,
):
    y_fn = cdist(np.expand_dims(fn0, axis=1),
                 np.expand_dims(fn1, axis=1), err_rel)
    y_zeta = cdist(np.expand_dims(zeta0, axis=1),
                   np.expand_dims(zeta1, axis=1), err_rel)
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


def _covariance_matrix(data: np.ndarray, n_block_rows: int) -> np.ndarray:
    """Compute covariance matrices. Only return positive time lags.

    Parameters
    ----------
    data : np.ndarray
        A numpy array of size (n_samples x n_channels)
    n_block_rows : int
        The number of block rows for building the block Toeplitz matrix.

    Returns
    -------
    np.ndarray
        [description]
    """
    # 0. sanity checks
    n_samples = data.shape[0]
    n_channels = data.shape[1]
    assert type(data) == np.ndarray, "numpy array expected"
    assert n_channels < n_samples, "expected more samples than channels"

    # 1. initalize covariance matrices
    R = np.zeros([n_channels, n_channels, n_block_rows * 2], dtype=data.dtype)
    for i in range(n_block_rows * 2):
        if i == 0:
            R[:, :, i] = np.dot(data.T[:, :], data[:, :]) / n_samples
        else:
            R[:, :, i] = np.dot(data.T[:, :-i], data[i:, :]) / n_samples
    return R


def _block_toeplitz(covarianceMatrices: np.ndarray, n_block_rows=None) -> np.ndarray:
    """Return a block Toeplitz matrix for a given set of covariance matrices."""
    # 0. sanity checks
    n_channels = covarianceMatrices.shape[0]
    n_samples = covarianceMatrices.shape[2]
    assert n_channels < n_samples, "expected more samples than channels"
    assert n_channels == covarianceMatrices.shape[1], "expected equal number of channels"

    # 1. construct block Toeplitz matrix
    if not n_block_rows:
        # if n_samples % 2 == 0:  # n_samples is an even number
        #     i = n_samples // 2
        # elif n_samples % 2 == 1:  # n_samples is an odd number
        #     i = n_samples // 2
        i = 20
    else:
        i = n_block_rows // 2

    T = np.zeros([n_channels * i, n_channels * i])
    for jBlockRow in range(i):
        row = covarianceMatrices[:, :, jBlockRow + i:jBlockRow:-1].reshape(
            [n_channels, n_channels * i], order='F')
        T[jBlockRow * n_channels:(jBlockRow + 1) * n_channels, :] = row
    return T


def _recover_system_matrices(U, S, V, n_channels, model_order=None):
    if not model_order:  # if no model_order has been defined
        S_integral = np.cumsum(S)
        # cut-off at 99% of integral -> determine rank
        cutoff = S_integral[-1] * 0.999
        model_order = S_integral[S_integral < cutoff].shape[0]
    if model_order == 0:
        return None, None
    U_1 = U[:, :model_order]
    S_1 = S[:model_order]
    # V_1 = V[:model_order, :model_order]

    # observability matrix:
    Observ = U_1 @ np.diag(np.sqrt(S_1))
    C = Observ[:n_channels, :]
    # TODO: fraglich, ob korrekt!!
    A = np.linalg.pinv(Observ[:-n_channels, :]) @ Observ[n_channels:, :]

    # controlability matrix: (not needed for modal ID)
    # Gamma = np.diag(np.sqrt(S_1)) @ V_1.T
    return A, C


def _modal_ID_from_system_matrices(A: np.ndarray, C: np.ndarray, dt: float):
    w, V = np.linalg.eig(A)
    # Eigenvalues in continuous time:
    omega = np.log(w) / dt
    # real frequency as the undamped frequency of the system
    freq = np.abs(omega) / 2 / np.pi
    xi = - np.real(omega) / np.abs(omega)
    # Modeshapes
    Psi = C @ V
    return freq, xi, Psi.T


def ssi_cov_poles(
    data: np.ndarray,
    fs: float,
    n_block_rows: int,
    max_model_order: int,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    """Calculate all poles via the SSI covariannce method up to a maximum model order.

    Parameters
    ----------
    data : np.ndarray
        A numpy array of size (n_samples x n_channels)
    fs : float
        The sampling frequency of the data.
    n_block_rows : int
        The number of block rows for building the block Toeplitz matrix.
    max_model_order : int
        The maximum model order as parameter for the SSI, for up to which the poles should be identified.

    Returns
    -------
    freq: List[np.ndarray]
        A list of 1-dimensional (model_order x 1) arrays containing the undamped eigenfrequencies of the identified poles.
    xi: List[np.ndarray]
        A list of 1-dimensional (model_order x 1) arrays containing the damping ratios of the identified poles.
    Psi: List[np.ndarray]
        A list of 2-dimensional (model_order x n_channels) arrays containing the modeshapes of the identified poles.

    """
    # 0. sanity checks
    n_channels = data.shape[1]
    n_samples = data.shape[0]
    assert type(data) == np.ndarray, "numpy array expected"
    assert n_channels < n_samples, "expected more samples than channels"
    assert n_block_rows >= max_model_order, "n_block_rows must be greater or equal than maxmodel_order"
    dt = 1 / fs

    # 1. build covariance matrices from data
    R = _covariance_matrix(data, n_block_rows)

    # 2. construct block toeplitz matrix
    T = _block_toeplitz(R, n_block_rows=n_block_rows)

    # 3. SVD of block toeplitz matrix
    U_full, S_full, V_full = np.linalg.svd(T)

    # # 4. determine maxmodel_order, if not given
    # if not maxmodel_order:  # if no maxmodel_order has been defined
    #     S_integral = cumtrapz(S_full)
    #     # cut-off at 99% of integral -> determine rank
    #     cutoff = S_integral[-1] * 0.99
    #     maxmodel_order = S_integral[S_integral < cutoff].shape[0] + 1

    # 5. loop over model Orders:
    freq = []
    xi = []
    Psi = []
    for i_order in range(1, max_model_order):
        # 5.a. recover A,C based on SVD and model order
        A, C = _recover_system_matrices(
            U_full, S_full, V_full, n_channels, model_order=i_order)

        # 5.b. modal ID based on system matrices
        freq_temp, xi_temp, Psi_temp = _modal_ID_from_system_matrices(A, C, dt)
        freq.append(freq_temp)
        xi.append(xi_temp)
        Psi.append(Psi_temp)

    return freq, xi, Psi


def ssi_cov_poles_for_model_order(
    data: np.ndarray,
    fs: float,
    n_block_rows: int,
    model_order: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Calculate all poles via the SSI covariannce method for a specified model order.

    Parameters
    ----------
    data : np.ndarray
        A numpy array of size (n_samples x n_channels)
    fs : float
        The sampling frequency of the data.
    n_block_rows : int
        The number of block rows for building the block Toeplitz matrix.
    model_order : int
        The model order as parameter for the SSI, for which the poles should be identified.


    Returns
    -------
    freq: np.ndarray
        An 1-dimensional (model_order x 1) array containing the undamped eigenfrequencies of the identified poles.
    xi: np.ndarray
        An 1-dimensional (model_order x 1) array containing the damping ratios of the identified poles.
    Psi: np.ndarray
        A 2-dimensional (model_order x n_channels) array containing the modeshapes of the identified poles.

    """
    # 0. sanity checks
    n_samples = data.shape[0]
    n_channels = data.shape[1]
    assert type(data) == np.ndarray, "numpy array expected"
    assert n_channels < n_samples, "expected more samples than channels"
    assert n_block_rows >= model_order, "n_block_rows must be greater or equal than model_order"

    # 1. build covariance matrices from data
    R = _covariance_matrix(data, n_block_rows)

    # 2. construct block toeplitz matrix
    T = _block_toeplitz(R, n_block_rows=n_block_rows)

    # 3. SVD of block toeplitz matrix
    U_full, S_full, V_full = np.linalg.svd(T)

    # 4. recover A,C based on SVD and model order
    A, C = _recover_system_matrices(
        U_full, S_full, V_full, n_channels, model_order=model_order)

    # 5. modal ID based on system matrices
    dt = 1 / fs
    freq, xi, Psi = _modal_ID_from_system_matrices(A, C, dt)
    return freq, xi, Psi


# TODO: This does not yield participation for model orders but for something different!!
def ssi_cov_model_order_participation(
    data: np.ndarray,
    fs: float,
    n_block_rows: int,
    max_model_order: int,
) -> np.ndarray:
    """Determine the cumulative participations of the model orders.

    Parameters
    ----------
    data : np.ndarray
        A numpy array of size (n_samples x n_channels)
    fs : float
        The sampling frequency of the data.
    n_block_rows : int
        The number of block rows for building the block Toeplitz matrix.
    max_model_order : int
        The maximum model order as parameter for the SSI, for up to which the participations are calculated.

    Returns
    -------
    participations: np.ndarray
        The normalized and added up eigenvalues of the block Toeplitz matrix.

    """
    # 0. sanity checks
    n_channels = data.shape[1]
    n_samples = data.shape[0]
    assert type(data) == np.ndarray, "numpy array expected"
    assert n_channels < n_samples, "expected more samples than channels"
    assert n_block_rows >= max_model_order, "n_block_rows must be greater or equal than maxmodel_order"

    # 1. build covariance matrices from data
    R = _covariance_matrix(data, n_block_rows)

    # 2. construct block toeplitz matrix
    T = _block_toeplitz(R, n_block_rows=n_block_rows)

    # 3. SVD of block toeplitz matrix
    U_full, S_full, V_full = np.linalg.svd(T)

    # 4. determine participations of system dofs
    # TODO: this is not scaled to model orders. DEBUG!!
    S_integral = np.cumsum(S_full)

    return S_integral / S_integral[-1]


def filter_ssi_single_order(
    freq: np.ndarray,
    xi: np.ndarray,
    Psi: np.ndarray,
    pairwise_occurence: bool = True,
    positive_damping: bool = True,
    max_damping: float = 0.15,
) -> np.ndarray:
    """Filter a single ssi model order based on acceptrance criteria.

    Parameters
    ----------
    freq: np.ndarray
        An 1-dimensional (model_order x 1) array containing the undamped eigenfrequencies of the identified poles.
    xi: np.ndarray
        An 1-dimensional (model_order x 1) array containing the damping ratios of the identified poles.
    Psi: np.ndarray
        A 2-dimensional (model_order x n_channels) array containing the modeshapes of the identified poles.
    pairwise_occurence : bool, optional
        If True, only return the first pole of poles that appear pairwise, by default True
    positive_damping : bool, optional
        If True, only return poles with positive damping, by default True
    max_damping : float, optional
        If specified, only return poles with damping below this threshold, by default 0.15

    Returns
    -------
    np.ndarray
        An 1-dimensional (model_order x 1) array boolean mask with the filtered poles.
    """
    # Filter a single ssi model order based on acceptrance criteria.
    # Parameters
    # ----------
    # freq: np.ndarray
    #     An 1-dimensional (model_order x 1) array containing the undamped eigenfrequencies of the identified poles.
    # xi: np.ndarray
    #     An 1-dimensional (model_order x 1) array containing the damping ratios of the identified poles.
    # Psi: np.ndarray
    #     A 2-dimensional (model_order x n_channels) array containing the modeshapes of the identified poles.

    # 0. determine parameters
    assert freq.ndim == xi.ndim == 1
    assert Psi.ndim == 2
    assert freq.shape == xi.shape
    # n_poles = freq.shape[0]

    # 1. filter for twin occurence, return only one of the twin poles
    if pairwise_occurence:
        filter_pairwise = np.zeros_like(freq, dtype=bool)

        # np.unique yields the unique values, the first occurence of such a unique value
        # and the number of occurences
        u_vals, u_idx, u_counts = np.unique(freq, return_index=True, return_counts=True)

        # The filter is true for all first occurences of unique values:
        filter_pairwise[u_idx] = True

        # The values that don't have a duplicate
        non_duplicate_values = u_vals[u_counts != 2]

        # The indices of these values are found with:
        non_duplicate_idx = np.where(np.isin(freq, non_duplicate_values))

        # They are not to be included in the result set:
        filter_pairwise[non_duplicate_idx] = False

    else:
        filter_pairwise = np.ones_like(freq, dtype=bool)

    # 2. filter for positive damping
    if positive_damping:
        filter_positive_damping = xi > 0
    else:
        filter_positive_damping = np.ones_like(freq, dtype=bool)

    # 3. filter for maximal allowed damping
    filter_max_damping = xi <= max_damping

    # 4. return all conditions applied
    return filter_pairwise & filter_positive_damping & filter_max_damping
