import numpy as np
import meshzoo
import pytest

import trimem.core as m

from scipy.sparse import coo_matrix
from scipy.spatial import KDTree


# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module", params=["cell-list", "verlet-list"])
def data(request):
    """Test mesh."""

    points, cells = meshzoo.rectangle_tri(
        np.linspace(0.0, 1.0, 20),
        np.linspace(0.0, 1.0, 20),
        variant="zigzag"
    )
    x = np.append(points, np.zeros((len(points),1)), axis=1)
    mesh = m.TriMesh(x, cells)

    class Data:
        pass

    d = Data()
    d.mesh  = mesh
    d.ltype = request.param

    return d

@pytest.fixture(params=[(0,44),(1,36),(2,20)])
def excl(request):
    """Exclusion level with result for num neighbours of node 151.

    params is list of (exclusion_level, num_neighbours_result)-tuples
    """
    class Data:
        pass
    d = Data()
    d.excl = request.param[0]
    d.nn   = request.param[1]
    return d

def get_nlist(mesh, ltype, rlist, excl):
    """Get a neighbour list."""

    params = m.EnergyParams()
    params.repulse_params.n_search        = ltype
    params.repulse_params.rlist           = rlist
    params.repulse_params.exclusion_level = excl

    return m.make_nlist(mesh, params)

# -----------------------------------------------------------------------------
#                                                                       test --
# -----------------------------------------------------------------------------
def test_distance_matrix(data):
    """Verify distance matrix againt kdtee implementation."""

    mesh = data.mesh
    x = mesh.points()

    nl = get_nlist(mesh, data.ltype, 0.2, 0)

    # compute distance matrix
    d,i,j = nl.distance_matrix(mesh, 0.123)
    A = coo_matrix((d,(i,j)), shape=(len(x),len(x)))
    M = A + A.T # kdtree gives full matrix

    # compare against kd-tree distance computation
    tree = KDTree(x)
    C = tree.sparse_distance_matrix(tree, 0.123)

    assert (C-M).max() == 0.0

def test_exclusion(data, excl):
    """Test neighbour lists exclusion level."""

    mesh = data.mesh
    x = mesh.points()

    nl = get_nlist(mesh, data.ltype, 0.2, excl.excl)

    # compute point distances for test point 151
    _,jdx = nl.point_distances(mesh, 151, 0.2)

    assert len(jdx) == excl.nn
