import helfrich as m
import helfrich.openmesh as om
import meshzoo
import numpy as np

import pytest

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
def get_mesh():
    """Get a mesh."""
    points, cells = meshzoo.icosa_sphere(8)
    return om.TriMesh(points, cells)

def get_energy_manager(mesh, bond_type, kb, ka, kv, kc, kt):
    """Setup energy manager."""
    l = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    a = m.area(mesh)/mesh.n_faces();

    bparams = m.BondParams()
    bparams.type = bond_type
    bparams.lc0  = 1.15*l
    bparams.lc1  = 0.85*l
    bparams.a0   = a

    eparams = m.EnergyParams()
    eparams.kappa_b        = kb
    eparams.kappa_a        = ka
    eparams.kappa_v        = kv
    eparams.kappa_c        = kc
    eparams.kappa_t        = kt
    eparams.area_frac      = 0.8
    eparams.volume_frac    = 0.8
    eparams.curvature_frac = 0.8
    eparams.bond_params    = bparams

    return m.EnergyManager(mesh, eparams)

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session", params=[m.BondType.Edge, m.BondType.Area])
def bond_type(request):
    """Parametrize for bond type."""
    class D:
        pass

    d = D()
    d.bond_type = request.param

    yield d

@pytest.fixture(scope="session", params=[(1.0, 1.0, 1.0, 1.0, 1.0)])
def data(bond_type, request):
    """Load this tests data"""

    mesh     = get_mesh()
    estore   = get_energy_manager(mesh,
                                  bond_type.bond_type,
                                  request.param[0],
                                  request.param[1],
                                  request.param[2],
                                  request.param[3],
                                  request.param[4])

    class Data:
        pass

    d = Data()
    d.mesh   = mesh
    d.estore = estore

    yield d

# -----------------------------------------------------------------------------
#                                                                      tests --
# -----------------------------------------------------------------------------
def test_gradient(data):
    """Test gradient."""

    mesh   = data.mesh
    estore = data.estore

    # finite difference gradient
    ref_grad = np.empty((mesh.n_vertices(),3))
    m.gradient(mesh, estore, ref_grad, 1.0e-8)

    # analytic gradient
    grad = estore.gradient()

    assert np.linalg.norm(grad - ref_grad)/np.linalg.norm(ref_grad) < 1.0e-4
