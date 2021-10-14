import numpy as np
import matplotlib.pyplot as plt
from pyomac import fdd, peak_picking, ssi_cov
from pyomac.plot import ssi_stability_plot, fdd_peak_picking_plot


def main():

    method = 'ssi'  # 'ssi' or 'fdd'

    with open('sample.csv') as csv_file:
        data = np.loadtxt(csv_file, skiprows=1, delimiter=',')
        # data.shape: (n_samples x n_channels) = (90000 x 18) 

    fs = 100  # sampling frequency [Hz]
    fmax = 25.

    if method == 'ssi':
        fn, ms, zeta, poles_stab, order_stab, poles, order = ssi_cov(data, fs, f_max=fmax, return_poles=True)
        fig, ax = ssi_stability_plot(data, fs, fn, poles_stab, order_stab, poles, order)

    elif method == 'fdd':
        f, s, u = fdd(data, fs, fmax=25)
        fn, ms = peak_picking(f, s, u, n_sv=2, thres=0.15, min_dist=0.5)

        fdd_peak_picking_plot(f, s, fn, n_sv=2, semilogy=True)
    else:
        # ToDo: Throw error
        pass

    # mode shape plots
    idx_n = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    idx_s = [3, 4, 5, 9, 10, 11, 15, 16, 17]

    for idx, val in enumerate(fn)[:3]:
        fig, ax = plt.subplots()

        ax.plot(ms[idx, idx_n])
        ax.plot(ms[idx, idx_s])
        ax.text(ax.get_xlim()[0] + 0.2, ax.get_ylim()[0] + 0.2, str(np.round(val, 2)) + ' Hz')
        plt.show()


if __name__ == "__main__":
    main()
