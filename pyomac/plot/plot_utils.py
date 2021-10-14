"""Plotting capabilities for pyomac."""
from typing import Tuple, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from scipy import signal


# def ssi_stability_plot(data, fs, fn, poles_stab, order_stab, poles, order, fmax=25, nperseg=2 ** 12, fig_obj=None):
#     ...


def ssi_stability_plot(
    poles: Sequence[np.ndarray],
    fmax: float = 25,
    model_orders: Optional[Sequence[int]] = None,
    fig_obj: Optional[Tuple[Figure, Axes]] = None,
    label: Optional[str] = None,
) -> Tuple[Figure, Axes]:
    """Plot the stability diagramm of poles over model order.

    Parameters
    ----------
    poles : Sequence[np.ndarray]
        Sequence of arrays containing the frequencies of the identified poles.
    fmax : float, optional
        maximum frequency to be displayed, by default 25
    model_orders : Optional[Sequence[int]], optional
        If supplied, a Sequence of model orders corresponding to the poles
    fig_obj : Optional[Tuple[Figure, Axes]], optional
        A tuple containing a matplotlib figure and axes to be drawn upon.
    label : Optional[str], optional
        The label attached to the scatter path collection.

    Returns
    -------
    Tuple[Figure, Axes]
        A tuple containing the matplotlib figure and axes.
    """
    # 0. check if fig, ax was supplied, else create it.
    if fig_obj:
        fig, ax = fig_obj

    else:
        fig, ax = plt.subplots()

    # 1. if model orders are not supplied, assume model orders starting from 1:
    if not model_orders:
        n_model_orders = len(poles)
        model_orders = range(1, n_model_orders + 1)

    # 2. concatenate all poles and corresponding model orders into single arrays,
    # which is way faster for plotting.
    all_poles = np.concatenate(poles)
    all_model_orders = np.concatenate([np.full_like(poles_order, i_order) for poles_order, i_order in zip(poles, model_orders)])

    # 3. plot
    ax.scatter(all_poles, all_model_orders, label=label)
    # hard-coded styling: (NOT USED HERE)
    # sc1 = ax.scatter(all_poles, all_model_orders, facecolor='None',
    #                     edgecolor='tab:grey', alpha=0.75)

    # 4. set axes limits and labels
    ax.set_xlim(0., fmax)
    ax.set_ylabel('Model Order [-]')
    ax.set_xlabel('Frequency [Hz]')

    return fig, ax


def ssi_stability_plot_spectrum(
    data: np.ndarray,
    fs: float,
    poles: Sequence[np.ndarray],
    model_orders: Optional[Sequence[int]] = None,
    fmax: float = 25,
    nperseg: int = 2 ** 12,
    fig_obj: Optional[Tuple[Figure, Axes]] = None,
) -> Tuple[Figure, Axes]:

    if fig_obj:
        fig, ax = fig_obj

    else:
        fig, ax = plt.subplots()

    if data.shape[1] < data.shape[0]:
        data = data.T

    # get pxx
    f_x, pxx = signal.welch(data, fs, nperseg=nperseg)
    pxx = pxx.mean(axis=0)

    _, _ = ssi_stability_plot(
        poles=poles, fmax=fmax, model_orders=model_orders, fig_obj=(fig, ax))
    ax.set_xlim(0., fmax)

    # line_f = None

    # for f in fn:
    #     line_f = ax.axvline(f, c='k', linestyle='--', linewidth=2.5)

    ax_log = ax.twinx()
    spec, = ax_log.semilogy(f_x, pxx, c='tab:blue')

    ax_log.get_yaxis().set_visible(False)

    ax.set_ylabel('Model Order [-]')
    ax.set_xlabel('Frequency [Hz]')

    # fig.legend((sc1, sc2, spec, line_f),
    #            ('Pole', 'Stable Pole', 'Spectral Density', 'Natural Frequencies'),
    #            ncol=2, loc='upper right', bbox_to_anchor=(1., 1.))

    # fig.tight_layout(rect=[0., 0., 1., 0.9])

    # plt.show()

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
