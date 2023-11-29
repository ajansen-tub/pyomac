"""Clustering procedures for modes."""

from typing import List, Sequence, Tuple, Optional, Union, NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import AgglomerativeClustering
from matplotlib.axes import Axes
from matplotlib.figure import Figure

# Data type definitions:


class ModalSet(NamedTuple):
    """Represents a modal set.

    implicit convention:
    frequencies: np.ndarray (n_modes x 1)
    dampings: np.ndarray (n_modes x 1)
    modeshapes: np.ndarray (n_modes x n_dof)
    """

    frequencies: np.ndarray
    dampings: np.ndarray
    modeshapes: np.ndarray

    def __str__(self) -> str:
        assert (
            self.frequencies.shape[0]
            == self.dampings.shape[0]
            == self.modeshapes.shape[0]
        )
        return "ModalSet(n_modes = {n}, modeshapes_array: {s}, frequencies={f})".format(
            n=self.frequencies.shape[0], f=self.frequencies, s=self.modeshapes.shape
        )

    @property
    def n_modes(self):
        return self.frequencies.shape[0]


class IndexedModalSet(NamedTuple):
    """Represents a modal set.

    implicit convention:
    indices: np.ndarray (n_modes x 1)
    frequencies: np.ndarray (n_modes x 1)
    dampings: np.ndarray (n_modes x 1)
    modeshapes: np.ndarray (n_modes x n_dof)
    """

    indices: np.ndarray
    frequencies: np.ndarray
    dampings: np.ndarray
    modeshapes: np.ndarray

    def __str__(self) -> str:
        assert (
            self.frequencies.shape[0]
            == self.dampings.shape[0]
            == self.modeshapes.shape[0]
            == self.indices.shape[0]
        )
        return "IndexedModalSet(n_modes = {n}, modeshapes_array: {s}, indices={i}, frequencies={f})".format(
            n=self.frequencies.shape[0],
            f=self.frequencies,
            s=self.modeshapes.shape,
            i=self.indices,
        )

    @property
    def n_modes(self):
        return self.frequencies.shape[0]


GeneralizedModalSet = Union[ModalSet, IndexedModalSet]

# ModalSet = namedtuple("ModalSet", ("frequencies", "dampings", "modeshapes"))

# IndexedModalSet = namedtuple(
#     "IndexedModalSet", ("indices", "frequencies", "dampings", "modeshapes")
# )
# # implicit convention:
# # indices: np.ndarray (n_modes x 1)
# # frequencies: np.ndarray (n_modes x 1)
# # dampings: np.ndarray (n_modes x 1)
# # modeshapes: np.ndarray (n_modes x n_dof)


def indexed_modal_sets_from_sequence(
    sequence: Sequence[ModalSet],
) -> Tuple[IndexedModalSet, ...]:
    """Convert a Sequence of ModalSets to IndexedModalSets, based on their item number.

    Parameters
    ----------
    sequence : Sequence[ModalSet]
        [description]

    Returns
    -------
    Tuple[IndexedModalSet, ...]
        [description]
    """
    return tuple(
        IndexedModalSet(
            indices=np.full_like(modal_set.frequencies, fill_value=i, dtype=int),
            frequencies=modal_set.frequencies,
            dampings=modal_set.dampings,
            modeshapes=modal_set.modeshapes,
        )
        for i, modal_set in enumerate(sequence)
    )


def modal_sets_from_lists(
    frequencies: List[np.ndarray],
    dampings: List[np.ndarray],
    modeshapes: List[np.ndarray],
) -> Tuple[ModalSet, ...]:
    """Combine a Tuple of ModalSets from results of ssi_cov_poles mehods.

    Parameters
    ----------
    frequencies: List[np.ndarray]
        A list of 1-dimensional (model_order x 1) arrays containing the undamped eigenfrequencies of the identified poles.
    dampings: List[np.ndarray]
        A list of 1-dimensional (model_order x 1) arrays containing the damping ratios of the identified poles.
    modeshapes: List[np.ndarray]
        A list of 2-dimensional (model_order x n_channels) arrays containing the modeshapes of the identified poles.

    Returns
    -------
    Tuple[ModalSet, ...]
        [description]
    """
    return tuple(
        ModalSet(f, d, P) for f, d, P in zip(frequencies, dampings, modeshapes)
    )


def _MAC(phi_1: np.ndarray, phi_2: np.ndarray) -> float:
    """Calculate the MAC value of two (complex) modeshape vectors phi_1 and phi_2.

    Arguments:
        phi_1 {np.ndarray} -- Modeshape 1
        phi_2 {np.ndarray} -- Modeshape 2

    Returns:
        float -- MAC Value
    """
    # assert phi_1.shape == phi_2.shape, "Expected two vectors of the same shape."
    return (
        np.abs(np.vdot(phi_1, phi_2)) / (np.vdot(phi_1, phi_1) * np.vdot(phi_2, phi_2))
    ).real


def _autoMAC(modeshapes: np.ndarray) -> np.ndarray:
    """Calculate the autoMAC values for a set of modeshapes.

    Parameters
    ----------
    modeshapes : np.ndarray
        A 2-dimensional (n_modes x n_dof) array containing the set of modeshapes

    Returns
    -------
    np.ndarray
        A 2-dimensional (n_modes x n_modes) array representing the MAC matrix
    """
    return MAC_matrix(modeshapes, modeshapes)


def MAC_matrix(phi_1: np.ndarray, phi_2: np.ndarray) -> np.ndarray:
    """Calculate the MAC values for two sets of modes.

    Parameters
    ----------
    phi_1 : np.ndarray
        A 2-dimensional (n_modes_1 x n_dof) array containing the first set of modeshapes
    phi_2 : np.ndarray
        A 2-dimensional (n_modes_2 x n_dof) array containing the second set of modeshapes

    Returns
    -------
    np.ndarray
        A 2-dimensional (n_modes_1 x n_modes_2) array representing the MAC matrix.

    Raises
    ------
    ValueError
        This method is only implemented for the sets of modes either both complex or both real.
    """
    assert phi_1.shape[1] == phi_2.shape[1], "n_dof of modeshapes have to match."

    # discern between real and complex modes
    # numpy.vdot cannot be used here, since
    # "it should only be used for vectors."
    # https://numpy.org/doc/stable/reference/generated/numpy.vdot.html
    if np.iscomplexobj(phi_1) or np.iscomplexobj(phi_2):
        return np.square(
            np.abs(np.dot(phi_1, phi_2.conj().T))
            / (
                np.linalg.norm(phi_1, axis=1)[:, np.newaxis]
                * np.linalg.norm(phi_2, axis=1)
            )
        )
    elif np.isrealobj(phi_1) and np.isrealobj(phi_2):
        return np.square(
            np.abs(np.dot(phi_1, phi_2.T))
            / (
                np.linalg.norm(phi_1, axis=1)[:, np.newaxis]
                * np.linalg.norm(phi_2, axis=1)
            )
        )
    else:
        raise ValueError("Only implemented for complex or real modes")


def _distance_matrix(
    all_freq: np.ndarray, all_Psi: np.ndarray, modeshape_weight: float = 1
) -> np.ndarray:
    """Create a distance matrix for clustering based on modal frequencies and shapes.

    This distance metric is based on:
    REYNDERS, E., J. HOUBRECHTS AND G. DE ROECK
    Fully automated (operational) modal analysis.
    Mechanical Systems and Signal Processing, 2012, 29, 228-250.

    Parameters
    ----------
    all_freq : np.ndarray
        A 1-dimensional (n_modes x 1) array containing the undamped eigenfrequencies
    all_Psi : np.ndarray
        A 2-dimensional (n_modes x n_dof) array containing the modeshapes

    Returns
    -------
    np.ndarray
        A 2-dimensional (n_modes x n_modes) array representing the distance matrix
    """
    # 0) assertions
    assert all_freq.shape[0] == all_Psi.shape[0]

    # 1) frequency distances and autoMAC
    distance_frequencies = np.abs(
        all_freq[:, np.newaxis] - all_freq[np.newaxis, :]
    ) / np.fmax(all_freq[:, np.newaxis], all_freq[np.newaxis, :])
    distance_MAC = _autoMAC(all_Psi)

    # 3) distance matrix
    distanceMatrix = distance_frequencies + modeshape_weight * (1 - distance_MAC)
    return np.around(distanceMatrix, decimals=10)


def _hierarchical_clustering(
    freq: Sequence[np.ndarray],
    Psi: Sequence[np.ndarray],
    distance_threshold: float = 0.2,
    linkage: str = "single",
    modeshape_weight: float = 1,
) -> AgglomerativeClustering:
    """Perform agglomerative (hierarchical) clustering on sequences of frequencies and modeshapes.

    Parameters
    ----------
    freq: Sequence[np.ndarray]
        A sequence of 1-dimensional (model_order x 1) arrays containing the undamped eigenfrequencies of the identified poles.
    Psi: Sequence[np.ndarray]
        A sequence of 2-dimensional (model_order x n_channels) arrays containing the modeshapes of the identified poles.

    distance_threshold : float, optional
        [description], by default 0.2

    Returns
    -------
    AgglomerativeClustering
        sklearn cluster object for further processing
    """
    # 0) assertions
    assert len(freq) == len(Psi)

    # 1) flatten data structures to arrays
    all_freq = np.squeeze(np.concatenate(freq))  # squeeze prevents dimension errors
    all_Psi = np.concatenate(Psi, axis=0)

    # 2) precompute weights for distance-based clustering
    D = _distance_matrix(all_freq, all_Psi, modeshape_weight)

    # 3) clustering phase
    cluster = AgglomerativeClustering(
        n_clusters=None,
        affinity="precomputed",
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_full_tree=True,
    ).fit(D)
    return cluster


def modal_clusters(
    indexed_modal_sets: Sequence[IndexedModalSet],
    distance_threshold: float = 0.2,
    linkage: str = "single",
    modeshape_weight: float = 1,
) -> Tuple[IndexedModalSet, ...]:
    """Cluster an indexed Sequence of modes.

    The index may be method-specific parameter (e.g. the model order in SSI)
    or the index of a dataset in a sequence of datasets.

    Parameters
    ----------
    indexed_modal_sets = Sequence[IndexedModalSet]
        A Sequence of IndexedModalSet
    distance_threshold : float, optional
        [description], by default 0.2

    Returns
    -------
    Tuple[IndexedModalSet, ...]
        The sequence of
    """
    # 0) transformation, assertions
    index, freq, xi, Psi = zip(*indexed_modal_sets)
    assert len(index) == len(freq) == len(xi) == len(Psi)

    # 1) flatten data structures to arrays
    all_indices = np.concatenate(index)
    all_freq = np.concatenate(freq)
    all_xi = np.concatenate(xi)
    all_Psi = np.concatenate(Psi, axis=0)

    # 2) replicate the index of a set for every item of the set
    # list_of_indices = []
    # for i, f, x, P in zip(index, freq, xi, Psi):
    #     assert f.shape == x.shape
    #     assert f.shape[0] == P.shape[0]
    #     list_of_indices.append(np.full(f.shape, i))
    # all_indices = np.concatenate(list_of_indices)

    # 3) cluster based on frequencies and modeshapes:
    cluster = _hierarchical_clustering(
        freq=freq,
        Psi=Psi,
        distance_threshold=distance_threshold,
        linkage=linkage,
        modeshape_weight=modeshape_weight,
    )

    # 4) rearrange clusters into IndexedModalSets
    return tuple(
        IndexedModalSet(
            all_indices[cluster.labels_ == i],
            all_freq[cluster.labels_ == i],
            all_xi[cluster.labels_ == i],
            all_Psi[cluster.labels_ == i, :],
        )
        for i in range(cluster.n_clusters_)
    )


def single_set_statistics(
    modal_set: GeneralizedModalSet,
) -> Tuple[float, float, float, float, float, float]:
    """Calculate statistics of a ModalSet.

    Parameters
    ----------
    modal_set : ModalSet
        A Modal Set is per definition a
        namedtuple containing:
        frequencies (n_modes x 1)
        dampings (n_modes x 1)
        modeshapes (n_modes x n_dof)

    Returns
    -------
    Tuple[float, float, float, float, float, float]
        mean_freq, mean_xi, mean_MAC, stdev_freq, stdev_xi, stdev_MAC
    """
    complete_MAC = _autoMAC(modal_set.modeshapes)
    mean_freq = np.mean(modal_set.frequencies)
    mean_xi = np.mean(modal_set.dampings)
    stdev_freq = np.std(modal_set.frequencies)
    stdev_xi = np.std(modal_set.dampings)
    mean_MAC = np.mean(complete_MAC)
    stdev_MAC = np.std(complete_MAC)
    return mean_freq, mean_xi, mean_MAC, stdev_freq, stdev_xi, stdev_MAC


def filter_clusters(
    modal_sets: Sequence[IndexedModalSet],
    max_freq_cov: float = 0.01,
    min_MAC: float = 0.9,
    min_n_modes: int = 3,
    notch_filter_bands: Optional[Sequence[Tuple[float, float]]] = None,
) -> Tuple[IndexedModalSet, ...]:
    """Filter a sequence of IndexedModalSets based on their statistics.

    While this relies on IndexedModalSets as datastructure it is only
    useful for sets of modes representing clusters, not sets of modes representing
    results from ssi or consequtive monitored datasets.

    Parameters
    ----------
    modal_sets : Sequence[IndexedModalSet]
        [description]
    max_freq_cov : float, optional
        [description], by default 0.01
    min_MAC : float, optional
        [description], by default 0.9
    min_n_modes : int, optional
        [description], by default 3

    Returns
    -------
    Tuple[IndexedModalSet, ...]
        [description]
    """

    if notch_filter_bands is None:
        notch_filter_bands = [(0.0, 0.0)]
    def _filter_set_(set: IndexedModalSet) -> bool:
        mean_freq, _, mean_MAC, stdev_freq, _, _ = single_set_statistics(set)
        n_modes = set.frequencies.shape[0]
        if (
            (stdev_freq / mean_freq < max_freq_cov)
            and (mean_MAC > min_MAC)
            and (n_modes >= min_n_modes)
            and not any(lower_bound <= mean_freq <= upper_bound for lower_bound, upper_bound in notch_filter_bands)
        ):
            return True
        else:
            return False

    return tuple(filter(_filter_set_, modal_sets))


def plot_indexed_clusters(
    modal_clusters: Sequence[IndexedModalSet],
    fig_obj: Optional[Tuple[Figure, Axes]] = None,
    flip_axes: bool = False,
    sort: Optional[str] = None,
    filter: bool = False,
    max_freq_cov: float = 0.01,
    min_MAC: float = 0.9,
    min_n_modes: int = 3,
) -> Tuple[Figure, Axes]:
    """Plot clusters as frequencies over indices or vice versa.

    Parameters
    ----------
    modal_clusters : Sequence[IndexedModalSet]
        [description]
    fig_obj : Optional[Tuple[Figure, Axes]], optional
        [description], by default None
    flip_axes : bool, optional
        [description], by default False
    sort : Optional[str], optional
        If supplied can be "freq" or "num_modes" to sort the clusters accordingly, by default None
    filter : bool, optional
        If True, filter the clusters with the lateron specified parameters, by default False
    max_freq_cov : float, optional
        [description], by default 0.01
    min_MAC : float, optional
        [description], by default 0.9
    min_n_modes : int, optional
        [description], by default 3

    Returns
    -------
    Tuple[Figure, Axes]
        [description]

    Raises
    ------
    ValueError
        [description]
    """
    if fig_obj:
        fig, ax = fig_obj
    else:
        fig, ax = plt.subplots()

    # 1) optionally filter and sort the clusters:
    # type annotation:
    _clusters: Union[Sequence[IndexedModalSet], Tuple[IndexedModalSet, ...]]
    if filter:
        _clusters = filter_clusters(modal_clusters, max_freq_cov, min_MAC, min_n_modes)
    else:
        _clusters = modal_clusters

    # type annotation:
    _sorted_clusters: Union[
        List[IndexedModalSet], Tuple[IndexedModalSet, ...], Sequence[IndexedModalSet]
    ]

    if sort == "freq":

        def sort_function(set: IndexedModalSet):
            return single_set_statistics(set)[0]  # mean frequency

        _sorted_clusters = sorted(_clusters, key=sort_function)

    elif sort == "num_modes":

        def sort_function(set: IndexedModalSet):
            return set.frequencies.shape[0]

        # unknown mypy error
        _sorted_clusters = sorted(_clusters, key=sort_function, reverse=True)  # type: ignore

    elif sort is None:
        _sorted_clusters = _clusters
    else:
        raise ValueError(
            "sort has to be 'freq' or 'num' or None, but is: {}".format(sort)
        )

    for i_c, cluster in enumerate(_sorted_clusters):
        mf, _, mean_MAC, sf, _, _ = single_set_statistics(cluster)
        if flip_axes:
            ax.scatter(
                cluster.indices,
                cluster.frequencies,
                label=f"{mf:02.3f} +- {(sf/mf):.2%} Hz",
            )
        else:
            ax.scatter(
                cluster.frequencies,
                cluster.indices,
                label=f"{mf:02.3f} +- {(sf/mf):.2%} Hz",
            )

    return fig, ax


def plot_cluster_dendrogramm(cluster: AgglomerativeClustering):
    # plot the dendrogramm of the cluster object (low level debugging tool)
    ...
