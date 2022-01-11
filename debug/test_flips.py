import helfrich as m
import helfrich.openmesh as om
import meshzoo
import numpy as np
import time

from hilbertcurve.hilbertcurve import HilbertCurve

from util import get_energy_manager

def hilbert_sort(points, cells, p=6):
    """Sort points along hilbert curve of order p."""

    # compute coordinates in [2**p]**n hypercube
    coords = np.copy(points)
    coords -= np.min(coords, axis=0)
    coords /= np.max(coords, axis=0)
    coords *= 2**p - 1
    coords = np.round(coords)

    # compute distances on hilbert curve
    curve = HilbertCurve(p,coords.shape[1])
    dist = np.array(curve.distances_from_points(coords))

    # sort along curve
    idxs = np.argsort(dist)
    ridxs = np.argsort(idxs)
    npoints = points[idxs]
    ncells = np.apply_along_axis(lambda x: [ridxs[i] for i in x],1,cells)

    return npoints, ncells

def test_flips():
    """Test flips."""
    p, c = meshzoo.icosa_sphere(32)
#    p, c = hilbert_sort(p,c)
    mesh = om.TriMesh(p, c)

    estore = get_energy_manager(mesh, m.BondType.Edge,
                                10.0, 1.0e4, 1.0e4, 0.0, 0.0, 1.0,
                                1.0, 1.0, 1.0)
    estore.print_info()

    om.write_mesh("test0.stl", mesh)
    start = time.time()
    for i in range(10):
        #flips = m.flip(mesh, estore, 0.1)
        flips = m.pflip(mesh, estore, 0.1)
        #flips = estore.energy()
    dt = time.time() - start
    print("flipped {} of {} edges".format(flips, mesh.n_edges()))
    print("took {}s".format(dt))
    om.write_mesh("test1.stl", mesh)

    dum = estore.energy()
    estore.print_info()

if __name__ == "__main__":
    test_flips()
