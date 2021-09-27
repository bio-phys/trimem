import helfrich as m
import helfrich.openmesh as om
import meshzoo
import numpy as np

from util import get_energy_manager

def test_flips():
    """Test flips."""
    #p, c = meshzoo.uv_sphere(num_points_per_circle=40, num_circles=20)
    p, c = meshzoo.icosa_sphere(8)
    mesh = om.TriMesh(p, c)

    estore = get_energy_manager(mesh, m.BondType.Edge,
                                10.0, 1.0e4, 1.0e4, 0.0, 0.0,
                                1.0, 1.0, 1.0)
    estore.print_info()

    om.write_mesh("test0.stl", mesh)
    flips = m.flip(mesh, estore, 1.0)
    print("flipped {} of {} edges".format(flips, mesh.n_edges()))
    om.write_mesh("test1.stl", mesh)

    dum = estore.energy()
    estore.print_info()

if __name__ == "__main__":
    test_flips()
