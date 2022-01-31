import numpy as np
import matplotlib.pyplot as plt
from pyomac import fdd, peak_picking, ssi_cov, ssi_cov_poles
from pyomac.plot import ssi_stability_plot, fdd_peak_picking_plot


def main():

    method = 'fdd'  # 'ssi' or 'fdd'

    with open('sample.csv') as csv_file:
        data = np.loadtxt(csv_file, skiprows=1, delimiter=',')
        # data.shape: (n_samples x n_channels) = (90000 x 18) 

    fs = 100  # sampling frequency [Hz]
    fmax = 25.

    if method == 'ssi':
        # deprecated:
        # fn, ms, zeta, poles_stab, order_stab, poles, order = ssi_cov(data, fs, f_max=fmax, return_poles=True)
        # 
        # use new API instead
        fn, zeta, ms = ssi_cov_poles(data=data, fs=fs, n_block_rows=60, max_model_order=30)
        fig, ax = ssi_stability_plot(poles=fn, fmax=fmax)
        plt.show()

    elif method == 'fdd':
        f, s, u = fdd(data, fs, fmax=25)
        fn, ms = peak_picking(f, s, u, n_sv=2, thres=0.15, min_dist=0.5)

        fdd_peak_picking_plot(f, s, fn, n_sv=2, semilogy=True)
    else:
        # ToDo: Throw error
        pass

    # mode shape plots
    idx_north = [0, 1, 2, 6, 7, 8, 12, 13, 14]
    idx_south= [3, 4, 5, 9, 10, 11, 15, 16, 17]

    # show modes from the first 3 modes of the last model order
    # when ssi is used, the results contain doubple poles, so every other pole has to be skipped.
    # also, the datastructures returned from ssi and fdd have to be treated differently
    if method  == "ssi":
        modes_to_show = slice(0, 3, 2)
        fn_plot = np.sort(fn[-1])
        ms_plot = ms[-1][np.argsort(fn[-1]), :]
    elif method == "fdd":
        modes_to_show = slice(0, 3)
        # import IPython
        # IPython.embed()
        fn_plot = fn
        ms_plot = ms

    for idx, val in enumerate(fn_plot[modes_to_show]):
        fig, ax = plt.subplots()

        ax.plot(ms_plot[idx, idx_north])
        ax.plot(ms_plot[idx, idx_south])
        # ax.text(ax.get_xlim()[0] + 0.2, ax.get_ylim()[0] + 0.2, str(np.round(val, 2)) + ' Hz')
        ax.set(title=str(np.round(val, 2)) + ' Hz')
        plt.show()


if __name__ == "__main__":
    main()
