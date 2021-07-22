import helfrich as m
import helfrich.openmesh as om
import numpy as np
import meshzoo
from collections import namedtuple

import pytest


# -----------------------------------------------------------------------------
#                                                             test constants --
# -----------------------------------------------------------------------------
n   = 32      # discretization size
eps = 1.0e-2  # relative acceptance error

# -----------------------------------------------------------------------------
#                                                                       util --
# -----------------------------------------------------------------------------
def sphere(r, n):
    """Get sphere mesh with analytical reference values."""
    points, cells = meshzoo.icosa_sphere(n)
    mesh = om.TriMesh(points*r, cells)
    # mesh, area, vol, curv, bending
    return mesh, 4*np.pi*r**2, 4/3*np.pi*r**3, 4*np.pi*r, 8*np.pi

def tube(r, n):
    """Get tube mesh with analytical reference values."""
    points, cells = meshzoo.tube(length=1, radius=r, n=n)
    mesh = om.TriMesh(points, cells)
    # mesh, area, vol, curv, bending
    return mesh, 2*np.pi*r, 2/3*np.pi*r**2, np.pi, np.pi/r


TParams = namedtuple("TParams", "name radius")

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
@pytest.fixture(params=[TParams("sphere", 1.0), TParams("sphere", 0.5),
                        TParams("tube", 1.0), TParams("tube", 0.5)])
def data(request):
    """facet-pair with properties for testing."""

    fun = globals()[request.param.name]
    mesh, a, v, c, b = fun(float(request.param.radius),n)

    results = {}
    results["area"]      = a
    results["volume"]    = v
    results["curvature"] = c
    results["bending"]   = b

    class Data:
        pass

    d = Data()
    d.mesh = mesh
    d.ref  = results

    yield d

@pytest.fixture()
def params(data):
    """Energy parameters."""

    mesh = data.mesh

    l = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    params = m.BondParams()
    params.type = m.BondType.Edge
    params.r = 2
    params.lc0 = 1.15*l
    params.lc1 = 0.85*l
    params.a0   = m.area(mesh)/mesh.n_faces()

    eparams = m.EnergyParams()
    eparams.kappa_b        = 1.0
    eparams.kappa_a        = 1.0
    eparams.kappa_v        = 1.0
    eparams.kappa_c        = 1.0
    eparams.kappa_t        = 1.0
    eparams.area_frac      = 1.0
    eparams.volume_frac    = 1.0
    eparams.curvature_frac = 1.0
    eparams.bond_params    = params

    estore = m.EnergyManager(mesh, eparams)

    data.estore = estore

    yield data

# -----------------------------------------------------------------------------
#                                                                       test --
# -----------------------------------------------------------------------------
def test_properties(params):
    """Test values of global properties."""

    e = params.estore.energy()
    p = params.estore.properties

    ref = params.ref

    def cmp(a,b):
        return np.abs(a-b)/b

    assert cmp(p.area, ref["area"]) < eps
    assert cmp(p.volume, ref["volume"]) < eps
    assert cmp(p.curvature, ref["curvature"]) < eps
    assert cmp(p.bending, ref["bending"]) < eps
