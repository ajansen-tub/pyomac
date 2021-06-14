import matplotlib.pyplot as plt
from scipy import signal


def ssi_stability_plot(data, fs, fn, poles_stab, order_stab, poles, order, fmax=25, nperseg=2 ** 12, fig_obj=None):

    if fig_obj:
        fig, ax = fig_obj

    else:
        fig, ax = plt.subplots()

    if data.shape[1] < data.shape[0]:
        data = data.T

    # get pxx
    f_x, pxx = signal.welch(data, fs, nperseg=nperseg)
    pxx = pxx.mean(axis=0)

    # plot
    sc1 = ax.scatter(poles[0], order, facecolor='None', edgecolor='tab:grey', alpha=0.75)
    sc2 = ax.scatter(poles_stab[0], order_stab, marker='x', color='tab:red')
    ax.set_xlim(0., fmax)

    line_f = None

    for f in fn:
        line_f = ax.axvline(f, c='k', linestyle='--', linewidth=2.5)

    ax_log = ax.twinx()
    spec, = ax_log.semilogy(f_x, pxx, c='tab:blue')

    ax_log.get_yaxis().set_visible(False)

    ax.set_ylabel('Model Order [-]')
    ax.set_xlabel('Frequency [Hz]')

    fig.legend((sc1, sc2, spec, line_f),
               ('Pole', 'Stable Pole', 'Spectral Density', 'Natural Frequencies'),
               ncol=2, loc='upper right', bbox_to_anchor=(1., 1.))

    fig.tight_layout(rect=[0., 0., 1., 0.9])

    plt.show()

    return fig, ax


def fdd_peak_picking_plot(f, s, fn, n_sv=1, semilogy=False):

    s_max = s[:, 0].max()

    fig, ax = plt.subplots()

    for idx in range(n_sv):
        if semilogy:
            ax.semilogy(f, s[:, idx] / s_max, label=r'$s_1$')

        else:
            ax.plot(f, s[:, idx] / s_max, label=r'$s_1$')

    for f_vline in fn:
        line_f = ax.axvline(f_vline, c='k', linestyle='--', linewidth=2.5)

    ax.set_xlim((f.min(), f.max()))
    ax.set_xlabel(r'Frequency [Hz]')
    ax.set_ylabel(r'Singular Values [-]')

    ax.legend()
    fig.tight_layout()
    plt.show()
