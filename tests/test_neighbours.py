import numpy as np
import pytest

import trimem.core as m

from scipy.sparse import coo_matrix
from scipy.spatial import KDTree

from util import rect

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
@pytest.fixture(scope="module", params=["cell-list", "verlet-list"])
def data(request):
    """Test mesh."""

    points, cells = rect(20,20)
    mesh = m.TriMesh(points, cells)

    class Data:
        pass

    d = Data()
    d.mesh  = mesh
    d.ltype = request.param

    return d

@pytest.fixture(params=[(0,44),(1,38),(2,26)])
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

@pytest.fixture(params=[0.2, 0.5, 1.2])
def rlist(request):
    """Parametrize test on rlist."""
    return request.param

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
def test_distance_matrix(data, rlist):
    """Verify distance matrix againt kdtee implementation."""

    mesh = data.mesh
    x = mesh.points()

    nl = get_nlist(mesh, data.ltype, rlist, 0)

    # compute distance matrix
    d,i,j = nl.distance_matrix(mesh, 0.123)
    M = coo_matrix((d,(i,j)), shape=(len(x),len(x)))
    if data.ltype == "cell-list":
        M = M + M.T # kdtree and verlet-lists gives full matrix

    # compare against kd-tree distance computation
    tree = KDTree(x)
    C = tree.sparse_distance_matrix(tree, 0.123)

    assert np.allclose(C.todense(), M.todense())

def test_exclusion(data, excl):
    """Test neighbour lists exclusion level."""

    mesh = data.mesh
    x = mesh.points()

    nl = get_nlist(mesh, data.ltype, 0.2, excl.excl)

    # compute point distances for test point 151
    _,jdx = nl.point_distances(mesh, 151, 0.2)

    assert len(jdx) == excl.nn
