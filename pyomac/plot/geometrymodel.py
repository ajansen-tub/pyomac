import numpy as np


class GeometryModel:

    def __init__(self, nodes=None, node_ids=None, edges=None, edge_ids=None, quads=None, quad_ids=None):

        self.nodes = nodes
        self.node_ids = node_ids
        self.edges = edges
        self.edge_ids = edge_ids
        self.quads = quads
        self.quad_ids = quad_ids

    def read_node_csv(self, file):

        with open(file) as csv_file:
            csv_data = np.loadtxt(csv_file, skiprows=1, delimiter=';')

        self.nodes = csv_data[:, 1:]
        self.node_ids = {int(key): int(value) for key, value in zip(csv_data[:, 0], np.arange(csv_data.shape[0]))}

    def read_quad_csv(self, file):

        with open(file) as csv_file:
            csv_data = np.loadtxt(csv_file, skiprows=1, delimiter=';', dtype=int)

        quad_data = csv_data[:, 1:]
        quad_ids = {int(key): int(value) for key, value in zip(csv_data[:, 0], np.arange(csv_data.shape[0]))}

        # convert to node row number
        quads = np.zeros_like(quad_data)

        for idx_i in range(quad_data.shape[0]):
            for idx_j in range(quad_data.shape[1]):
                quads[idx_i, idx_j] = self.node_ids[quad_data[idx_i, idx_j]]

        self.quads = quads
        self.quad_ids = quad_ids

    def quads_to_vtk(self):

        cell_type = (np.ones((self.quads.shape[0])) * 9).astype(int)
        cells = np.hstack(np.concatenate((np.ones((self.quads.shape[0], 1)) * 4, self.quads), axis=1)).astype(int)

        return cells, cell_type
