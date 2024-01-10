"""Clustering procedures for modes."""
from typing import List, NamedTuple, Optional, Sequence, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from sklearn.cluster import AgglomerativeClustering

MAX_CHUNKSIZE_CLUSTERING = 512


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


def _distance_matrix_memory_efficient(
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

    # Precompute norm
    norm_Psi = np.linalg.norm(all_Psi, axis=1).astype(np.float32)

    # Convert inputs to float32 if they aren't already
    all_freq = all_freq.astype(np.float32)
    all_Psi = all_Psi.astype(np.float32)

    # MAC_matrix functionality integrated
    if np.iscomplexobj(all_Psi):
        mac_result = np.dot(all_Psi, all_Psi.conj().T).astype(np.float32)
    elif np.isrealobj(all_Psi):
        mac_result = np.dot(all_Psi, all_Psi.T).astype(np.float32)
    else:
        raise ValueError("Only implemented for complex or real modes")

    np.abs(mac_result, out=mac_result)
    mac_result /= norm_Psi[:, np.newaxis]
    mac_result /= norm_Psi
    np.square(mac_result, out=mac_result)

    # Frequency distances calculation
    freq_diff = all_freq[:, np.newaxis] - all_freq
    np.abs(freq_diff, out=freq_diff)
    max_freq = np.maximum(all_freq[:, np.newaxis], all_freq)
    freq_diff /= max_freq

    # Final distance matrix calculation
    distanceMatrix = freq_diff + modeshape_weight * (1 - mac_result)
    np.around(distanceMatrix, decimals=10, out=distanceMatrix)
    return distanceMatrix


def _compute_meta_distance_matrix(
    aggregated_clusters,
    modeshape_weight: float = 1,
    linkage: str = "single",
) -> np.ndarray:
    """
    Compute the meta distance matrix for a set of aggregated clusters.

    This function calculates the pairwise distances between each pair of aggregated clusters
    using a specified linkage method. The distance is computed based on the frequency and
    modeshape data of the clusters.

    Parameters
    ----------
    aggregated_clusters : List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples, where each tuple contains the frequency (first element) and
        modeshape (second element) data for an aggregated cluster.
    linkage : str
        The linkage criterion to use for calculating the distance between clusters.
        It should be compatible with the `_inter_cluster_distance` function.

    Returns
    -------
    np.ndarray
        A symmetric 2D array (matrix) where the element at [i, j] represents the distance
        between the i-th and j-th cluster. The matrix is of size `num_clusters x num_clusters`,
        where `num_clusters` is the number of aggregated clusters.

    Notes
    -----
    The function assumes that the distance between each pair of clusters is symmetric,
    i.e., the distance from cluster i to cluster j is the same as from cluster j to cluster i.
    The `_inter_cluster_distance` function is used to compute the distance between two clusters,
    which should be defined elsewhere in the code and should be compatible with the linkage method used.
    """
    num_clusters = len(aggregated_clusters)
    meta_distance_matrix = np.zeros((num_clusters, num_clusters), dtype=np.float64)
    for i in range(num_clusters):
        for j in range(i + 1, num_clusters):  # Matrix is symmetric
            dist = _inter_cluster_distance(
                aggregated_clusters[i][0],
                aggregated_clusters[j][0],
                aggregated_clusters[i][1],
                aggregated_clusters[j][1],
                modeshape_weight=modeshape_weight,
                linkage=linkage,
            )
            meta_distance_matrix[i, j] = dist
            meta_distance_matrix[j, i] = dist
    return meta_distance_matrix


def _inter_cluster_distance(
    freq_1: np.ndarray,
    freq_2: np.ndarray,
    Psi_1: np.ndarray,
    Psi_2: np.ndarray,
    modeshape_weight: float = 1,
    linkage: str = "single",
) -> np.ndarray:
    """
    Compute the distance between two clusters based on their frequency and modeshape data.

    This function calculates the distance between two clusters using different linkage criteria:
    'single', 'complete', 'average', or 'centroid'. The distance metric is based on the frequency
    (freq) and modeshape (Psi) data of the clusters.

    Parameters
    ----------
    freq_1 : np.ndarray
        An array of frequencies for the first cluster.
    freq_2 : np.ndarray
        An array of frequencies for the second cluster.
    Psi_1 : np.ndarray
        An array of modeshapes for the first cluster.
    Psi_2 : np.ndarray
        An array of modeshapes for the second cluster.
    modeshape_weight : float, optional
        A weight factor for the modeshapes in the distance calculation, by default 1.
    linkage : str, optional
        The linkage criterion to use for calculating the distance. It can be 'single', 'complete',
        'average', or 'centroid', by default 'single'.

    Returns
    -------
    np.ndarray
        The computed distance between the two clusters.

    Raises
    ------
    ValueError
        If an unknown linkage type is provided.

    Notes
    -----
    - 'single' linkage computes the minimum distance between any two points in each pair of clusters.
    - 'complete' linkage computes the maximum distance between any two points in each pair of clusters.
    - 'average' linkage computes the average distance between points in each pair of clusters.
    - 'centroid' linkage computes the distance between the centroids of the clusters, factoring in
      both frequency and modeshape data.

    The function assumes that the frequency data (freq_1, freq_2) and the modeshape data (Psi_1, Psi_2)
    correspond to each other in the context of the clusters being compared.
    """
    if linkage == "single":
        # For single linkage, compute the minimum distance between any two points in each pair of clusters
        return np.min(
            _inter_cluster_distance_matrix(
                freq_1, freq_2, Psi_1, Psi_2, modeshape_weight
            )
        )
    elif linkage == "complete":
        # For complete linkage, compute the maximum distance between any two points in each pair of clusters
        return np.max(
            _inter_cluster_distance_matrix(
                freq_1, freq_2, Psi_1, Psi_2, modeshape_weight
            )
        )
    elif linkage == "average":
        # For average linkage, compute the average distance between points in each pair of clusters
        return np.mean(
            _inter_cluster_distance_matrix(
                freq_1, freq_2, Psi_1, Psi_2, modeshape_weight
            )
        )
    elif linkage == "centroid":
        # For centroid linkage, compute the distance between the centroids of clusters
        mean_f_1 = np.mean(freq_1)
        mean_f_2 = np.mean(freq_2)
        mac_result = np.mean(MAC_matrix(Psi_1, Psi_2))
        dist_freq = np.abs(mean_f_1 - mean_f_2) / np.fmax(mean_f_1, mean_f_2)
        return dist_freq + modeshape_weight * (1 - mac_result)
    else:
        raise ValueError("Unknown linkage type")


def _inter_cluster_distance_matrix(
    freq_1: np.ndarray,
    freq_2: np.ndarray,
    Psi_1: np.ndarray,
    Psi_2: np.ndarray,
    modeshape_weight: float = 1,
) -> np.ndarray:
    """Create a distance matrix between two clusters based on modal frequencies and shapes.

    This distance metric is based on:
    REYNDERS, E., J. HOUBRECHTS AND G. DE ROECK
    Fully automated (operational) modal analysis.
    Mechanical Systems and Signal Processing, 2012, 29, 228-250.

    Parameters
    ----------
    freq_1 : np.ndarray
        A 1-dimensional (n_modes x 1) array containing the undamped eigenfrequencies
    freq_2 : np.ndarray
        A 1-dimensional (n_modes x 1) array containing the undamped eigenfrequencies
    Psi_1 : np.ndarray
        A 2-dimensional (n_modes x n_dof) array containing the modeshapes
    Psi_2 : np.ndarray
        A 2-dimensional (n_modes x n_dof) array containing the modeshapes

    Returns
    -------
    np.ndarray
        A 2-dimensional (n_modes x n_modes) array representing the distance matrix
    """
    # 0) assertions
    assert freq_1.shape[0] == Psi_1.shape[0]
    assert freq_2.shape[0] == Psi_2.shape[0]

    # Convert inputs to float32 if they aren't already
    freq_1 = freq_1.astype(np.float32)
    freq_2 = freq_2.astype(np.float32)
    Psi_1 = Psi_1.astype(np.float32)
    Psi_2 = Psi_2.astype(np.float32)

    mac_result = MAC_matrix(Psi_1, Psi_2).astype(np.float32)

    # Frequency distances calculation
    freq_diff = freq_1[:, np.newaxis] - freq_2
    np.abs(freq_diff, out=freq_diff)
    freq_diff /= np.fmax(freq_1[:, np.newaxis], freq_2[np.newaxis, :])

    # Final distance matrix calculation

    distanceMatrix = freq_diff + modeshape_weight * (1 - mac_result)
    np.around(distanceMatrix, decimals=10, out=distanceMatrix)
    return distanceMatrix


def _hierarchical_clustering_chunked(
    freq: Sequence[np.ndarray],
    Psi: Sequence[np.ndarray],
    chunk_size: int,
    distance_threshold: float = 0.2,
    linkage: str = "single",
    modeshape_weight: float = 1,
) -> List[np.ndarray]:
    """
    Perform divide-and-conquer hierarchical clustering on sequences of frequencies and modeshapes.

    Parameters
    ----------
    freq: Sequence[np.ndarray]
        A sequence of 1D arrays containing frequencies.
    Psi: Sequence[np.ndarray]
        A sequence of 2D arrays containing modeshapes.
    chunk_size: int
        The maximum size of each chunk for divide-and-conquer approach.
    distance_threshold: float, optional
        The threshold for clustering.
    linkage: str, optional
        The linkage criterion ('single', 'complete', 'average', 'centroid').
    modeshape_weight: float, optional
        A weight factor for the modeshapes in distance calculation.

    Returns
    -------
    List[np.ndarray]
        A list of cluster labels for each data point.
    """

    # Step 1: Chunking the Data
    chunks = _create_chunks(freq, Psi, chunk_size)

    # Step 2: Clustering Each Chunk Independently
    chunk_cluster_labels = [
        _cluster_chunk(chunk, distance_threshold, linkage, modeshape_weight)
        for chunk in chunks
    ]

    # Step 3: Merging Clusters Across Chunks
    final_labels = _merge_clusters(
        chunks=chunks, chunk_clusters=chunk_cluster_labels, linkage=linkage
    )

    return final_labels


def _create_chunks(
    freq: Sequence[np.ndarray],
    Psi: Sequence[np.ndarray],
    chunk_size: int,
    randomize: bool = True,
):
    """
    Divide the data into chunks.

    Parameters
    ----------
    freq: Sequence[np.ndarray]
        A sequence of 1D arrays containing frequencies.
    Psi: Sequence[np.ndarray]
        A sequence of 2D arrays containing modeshapes.
    chunk_size: int
        The maximum size of each chunk.
    randomize: bool
        Pick chunks at random

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples, each containing a chunk of freq and Psi.
    """

    # Ensure that freq and Psi have the same number of elements
    assert len(freq) == len(Psi), "The length of freq and Psi must be the same."

    # Flatten the freq and Psi arrays
    all_freq = np.concatenate(freq)
    all_Psi = np.concatenate(Psi, axis=0)

    # Ensure the concatenated dimensions are compatible
    assert (
        all_freq.shape[0] == all_Psi.shape[0]
    ), "Mismatch in number of elements between freq and Psi."

    total_samples = all_freq.shape[0]

    # Randomize if required
    if randomize:
        rand_indices = np.random.permutation(total_samples)
        all_freq = all_freq[rand_indices]
        all_Psi = all_Psi[rand_indices, :]

    # Create chunks
    chunks = []
    for start_idx in range(0, total_samples, chunk_size):
        end_idx = min(start_idx + chunk_size, total_samples)
        chunk_freq = all_freq[start_idx:end_idx]
        chunk_Psi = all_Psi[start_idx:end_idx, :]
        chunks.append((chunk_freq, chunk_Psi))

    return chunks


def _cluster_chunk(chunk, distance_threshold, linkage, modeshape_weight):
    """
    Cluster an individual chunk using the hierarchical clustering method.

    Parameters
    ----------
    chunk: Tuple[np.ndarray, np.ndarray]
        A tuple containing a chunk of freq and Psi data.
    distance_threshold: float
        The threshold for clustering.
    linkage: str
        The linkage criterion ('single', 'complete', 'average', 'centroid').
    modeshape_weight: float
        A weight factor for the modeshapes in distance calculation.

    Returns
    -------
    np.ndarray
        Cluster labels for this chunk.
    """
    freq_chunk, Psi_chunk = chunk
    # Ensure that the chunk data is in the correct format (sequence of arrays)
    freq_chunk = [freq_chunk]
    Psi_chunk = [Psi_chunk]

    sklearn_linkage = "average" if linkage == "centroid" else linkage

    # Apply hierarchical clustering to the chunk
    cluster = _hierarchical_clustering(
        freq=freq_chunk,
        Psi=Psi_chunk,
        distance_threshold=distance_threshold,
        linkage=sklearn_linkage,
        modeshape_weight=modeshape_weight,
    )

    # Return the cluster labels
    return cluster.labels_


def _merge_clusters(
    chunks,
    chunk_clusters,
    linkage: str = "single",
    distance_threshold: float = 0.2,
    modeshape_weight: float = 1,
):
    """
    Merge clusters from different chunks.

    Parameters
    ----------
    chunks: List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples containing chunks of freq and Psi data.
    chunk_clusters: List[np.ndarray]
        A list where each element contains the cluster labels for a chunk.
    linkage: str
        The linkage criterion ('single', 'complete', 'average', 'centroid').
    distance_threshold : float, optional
        [description], by default 0.2

    Returns
    -------
    np.ndarray
        The final cluster labels after merging.
    """

    # Aggregate cluster data from each chunk
    aggregated_clusters = _aggregate_cluster_data(chunks, chunk_clusters)

    # Compute the meta distance matrix between clusters
    meta_distance_matrix = _compute_meta_distance_matrix(
        aggregated_clusters,
        linkage=linkage,
        modeshape_weight=modeshape_weight,
    )

    sklearn_linkage = "average" if linkage == "centroid" else linkage

    # Apply hierarchical clustering to the meta distance matrix
    final_clustering = AgglomerativeClustering(
        affinity="precomputed",
        linkage=sklearn_linkage,
        n_clusters=None,
        distance_threshold=distance_threshold,
    )

    final_clustering.fit(meta_distance_matrix)

    # Map the cluster labels from final clustering back to the original data points
    final_labels = _map_labels_to_original_data(
        final_clustering.labels_, chunk_clusters
    )

    return final_labels


def _aggregate_cluster_data(chunks, chunk_clusters):
    """
    Aggregate the freq and Psi data for each cluster identified in each chunk.

    Parameters
    ----------
    chunks: List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples containing chunks of freq and Psi data.
    chunk_clusters: List[np.ndarray]
        A list where each element contains the cluster labels for a chunk.

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
        A list of tuples, each containing aggregated freq and Psi data for a cluster.
    """
    aggregated_clusters = []

    for (chunk_freq, chunk_Psi), labels in zip(chunks, chunk_clusters):
        unique_labels = np.unique(labels)

        for label in unique_labels:
            # Indices of data points in this cluster
            indices = np.where(labels == label)[0]

            # Aggregate freq and Psi data for this cluster
            aggregated_freq = chunk_freq[indices]
            aggregated_Psi = chunk_Psi[indices, :]

            aggregated_clusters.append((aggregated_freq, aggregated_Psi))

    return aggregated_clusters


def _map_labels_to_original_data(final_labels, chunk_clusters):
    """
    Map the final cluster labels from the merged clusters back to the original data points.

    Parameters
    ----------
    final_labels : np.ndarray
        The cluster labels obtained from the final clustering of the meta clusters.
    chunk_clusters : List[np.ndarray]
        A list where each element contains the cluster labels for a chunk.

    Returns
    -------
    np.ndarray
        An array of cluster labels for each original data point.
    """

    # Initialize an empty list to store the final mapped labels
    mapped_labels = []

    # Counter for the number of clusters already processed
    cluster_counter = 0

    # Iterate over each chunk's cluster labels
    for labels in chunk_clusters:
        # Map each chunk's cluster label to the final cluster label
        mapped_chunk_labels = [
            final_labels[cluster_counter + label] for label in labels
        ]
        mapped_labels.extend(mapped_chunk_labels)

        # Update the cluster counter by the number of unique clusters in this chunk
        cluster_counter += len(np.unique(labels))

    return np.array(mapped_labels)


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
    D = _distance_matrix_memory_efficient(all_freq, all_Psi, modeshape_weight)

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
    """
    Perform hierarchical clustering on a sequence of modal data sets.

    This function clusters modal data based on their frequencies and modeshapes. It supports
    both standard and chunked hierarchical clustering, depending on the size of the data. For
    larger datasets, a divide-and-conquer approach is used to manage computational complexity.

    Parameters
    ----------
    indexed_modal_sets : Sequence[IndexedModalSet]
        A sequence of IndexedModalSet objects. Each IndexedModalSet contains data related to
        a specific mode, including its index, frequency, damping ratio, and modeshape.
    distance_threshold : float, optional
        The threshold for clustering. Clusters are merged if their distance is below this
        threshold. Default is 0.2.
    linkage : str, optional
        The linkage criterion to use in hierarchical clustering. Options include 'single',
        'complete', 'average', and 'centroid'. Default is 'single'.
    modeshape_weight : float, optional
        A weight factor for the modeshapes in the distance calculation. Default is 1.

    Returns
    -------
    Tuple[IndexedModalSet, ...]
        A tuple of IndexedModalSet objects, each representing a cluster of modes. The modes
        in each cluster are similar based on their frequencies and modeshapes.

    Notes
    -----
    - The function decides whether to use standard or chunked clustering based on the total
      number of modes. If the number of modes exceeds a predefined threshold (MAX_CHUNKSIZE_CLUSTERING),
      chunked clustering is used.
    - In chunked clustering, the data is divided into smaller chunks, each of which is clustered
      independently. The clusters are then merged across chunks.
    - This function is particularly useful in modal analysis where clustering of modes is required
      to identify similar behavior or characteristics among different modes.
    """
    # 0) transformation, assertions
    index, freq, xi, Psi = zip(*indexed_modal_sets)
    assert len(index) == len(freq) == len(xi) == len(Psi)

    # 1) flatten data structures to arrays
    all_indices = np.concatenate(index)
    all_freq = np.concatenate(freq)
    all_xi = np.concatenate(xi)
    all_Psi = np.concatenate(Psi, axis=0)

    # 2) determine, if clustering needs to be processed chunk-wise
    n_total_modes = len(all_indices)
    if n_total_modes < MAX_CHUNKSIZE_CLUSTERING:
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
    else:
        # 3) cluster based on frequencies and modeshapes:
        cluster_labels = _hierarchical_clustering_chunked(
            freq=freq,
            Psi=Psi,
            chunk_size=MAX_CHUNKSIZE_CLUSTERING,
            distance_threshold=distance_threshold,
            linkage=linkage,
            modeshape_weight=modeshape_weight,
        )
        clusters_indices = np.unique(cluster_labels)

        # 4) rearrange clusters into IndexedModalSets
        return tuple(
            IndexedModalSet(
                all_indices[cluster_labels == i],
                all_freq[cluster_labels == i],
                all_xi[cluster_labels == i],
                all_Psi[cluster_labels == i, :],
            )
            for i in clusters_indices
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
            and not any(
                lower_bound <= mean_freq <= upper_bound
                for lower_bound, upper_bound in notch_filter_bands
            )
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
    index_offset: Optional[int] = 0,
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
    index_offset : int, optional
        offset index, by default 0

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
                cluster.indices + index_offset,
                cluster.frequencies,
                label=f"{mf:02.3f} +- {(sf/mf):.2%} Hz",
            )
        else:
            ax.scatter(
                cluster.frequencies,
                cluster.indices + index_offset,
                label=f"{mf:02.3f} +- {(sf/mf):.2%} Hz",
            )

    return fig, ax


def plot_cluster_dendrogramm(cluster: AgglomerativeClustering):
    # plot the dendrogramm of the cluster object (low level debugging tool)
    ...
