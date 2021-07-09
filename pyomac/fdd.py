import numpy as np
from scipy import signal
from peakutils import indexes


def fdd(data, fs: float, method: str = 'welch', nperseg: int = 2 ** 12, fmax: float = 20):  # zum Beispiel
    """
    Compute Frequency Domain Decomposition (FDD)

    Parameters
    ----------
    data : array_like
        Time series of measurement values
    fs : float
        Sampling frequency [Hz] of the `data` time series
    method : str, optional
        Determines which method is used to calculate the Cross Power Spectral Density (CPSD) matrix, either 'welch' or
        'direct'. Defaults to 'welch'
    nperseg : int, optional
        Length of each segment when calculating the CPSD using the Welch-Method. Defaults to 2**12
    fmax : float, optional
        Maximum frequency [Hz] for which singular value decomposition is obtained. Defaults to 20 Hz


    Returns
    -------
    fn : ndarray
        Identified natural frequencies [Hz]
    zeta : ndarray
        Identified modal damping [-]


    See Also
    --------
    pyomac.fdd.fdd_welch : Similar function in SciPy.


    References
    ----------
    .. [1] Brincker, R., Zhang, L., & Andersen, P. (2001). Modal identification of output-only systems using frequency domain decomposition. Smart materials and structures, 10(3), 441.

    """

    if method == 'welch':
        f, s, u = fdd_welch(data, fs, nperseg, fmax)
    elif method == 'direct':
        f, s, u = fdd_direct(data, fs)
    else:
        raise ValueError(f"'method has to be 'welch' or 'direct', but is '{method}'")

    return f, s, u


def fdd_welch(data, fs, nperseg=2**12, fmax=20):

    n_ch = data.shape[1]
    g_xx = np.zeros((int(nperseg / 2 + 1), n_ch, n_ch), dtype=complex)
    # TODO: check, if faster implementation is possible
    for i in range(n_ch):
        for j in range(n_ch):
            f, g_xx[:, j, i] = signal.csd(
                data[:, i], data[:, j], fs=fs, nperseg=nperseg, detrend='linear')

    # Bandwidth of interest in [Hz]
    df = f[1] - f[0]
    n_fmax = int(fmax / df)
    n_f_calculated = g_xx.shape[0]
    if n_fmax > n_f_calculated:
        raise ValueError(f'With fmax={fmax} and nperseg={nperseg}, {n_fmax} frequencies are of interest, but only {n_f_calculated} could be calculated.')

    f = f[:n_fmax]
    s = np.zeros((n_fmax, n_ch))
    u = np.zeros((n_fmax, n_ch, n_ch), dtype=complex)

    # TODO: SVD can be unified with fdd_direct
    for i in range(n_fmax):  # SVD of PSD matrix
        u[i, :, :], s[i], _ = np.linalg.svd(np.squeeze(g_xx[i, :, :]))

    return f, s, u


def peak_picking(f, s, u, n_sv=2, thres=0.1, min_dist=0.5, abs_max=True):

    if abs_max:
        max_s = s[:, 0].max()

    else:
        max_s = s.max(axis=0)

    f_peaks = np.empty((0,))
    ms = np.empty((0, u.shape[1]))

    for n in range(n_sv):
        if isinstance(max_s, float):
            thres_abs = max_s * thres

        if isinstance(max_s, np.ndarray):
            thres_abs = max_s[n] * thres

        idx_peaks = (indexes(s[:, n], thres=thres_abs, thres_abs=True, min_dist=int(min_dist / f[1])))

        f_peaks = np.concatenate((f_peaks, f[idx_peaks]))
        ms = np.concatenate((ms, np.real(u[idx_peaks, :, n])), axis=0)

    idx_sort = f_peaks.argsort()

    f_peaks = f_peaks[idx_sort]
    ms = ms[idx_sort]
    ms = ms / np.abs(ms).max(axis=1).reshape(-1, 1)

    return f_peaks, ms


def fdd_direct(data, fs):
    f, cpsd_matrix = cpsd_via_fft(data, fs)
    u, s, _ = np.linalg.svd(cpsd_matrix, compute_uv=True, hermitian=True)
    return f, s, u


def fft(data, fs):
    '''Perform FFT on data based on scan frequency.
    Return complex spectrums and frequencies'''
    nSamples = data.shape[0]
    NyquistFreq = int(np.floor(nSamples / 2 + 1))
    f = np.linspace(0, fs / 2, NyquistFreq, endpoint=True)
    data_fft = np.zeros(data.shape, dtype=complex)
    # TODO: check, if faster implementation is possible
    # for iChannel in range(nChannels):
    #     data_fft[:, iChannel] = np.fft.fft(data[:, iChannel])
    data_fft = np.fft.fft(data.T).T
    return f, data_fft


def cpsd_via_fft(data, fs):
    f, data_fft = fft(data, fs)
    nFrequencies = f.shape[0]
    nChannels = data_fft.shape[1]
    cpsd_matrix = np.zeros((nFrequencies, nChannels, nChannels), dtype=complex)
    # TODO: check, if faster implementation is possible.
    for iChannel in range(nChannels):
        for jChannel in range(nChannels):
            cpsd_matrix[:, iChannel, jChannel] = (
                data_fft[:, iChannel] * np.conj(data_fft[:, jChannel]))[:nFrequencies]
    return f, cpsd_matrix
