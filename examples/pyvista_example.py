import numpy as np
from pyomac.plot import GeometryModel
import pyvista as pv


def main():
    # model of structure
    geom = GeometryModel()

    geom.read_node_csv('nodes.csv')
    geom.read_quad_csv('quads.csv')

    # model for mode shapes
    # ToDo: add attribute sensors to GeometryModel

    sensors = np.array([1, 4])
    direction = np.zeros((sensors.shape[0], 3))
    direction[:, 2] = 1.
    ms = np.array([1., 0.5])

    # Plot mode shape using pyvista
    cells, cell_type = geom.quads_to_vtk()
    mesh = pv.UnstructuredGrid(cells, cell_type, geom.nodes)

    # plot using the plotting class
    pl = pv.Plotter()
    pl.add_mesh(mesh, show_edges=True)

    for idx, sensor in enumerate(sensors):
        pl.add_arrows(geom.nodes[sensor], direction=direction[idx], mag=5 * ms[idx], color='red')

    pl.background_color = 'white'
    pl.show_axes()
    pl.show()


if __name__ == "__main__":
    main()
