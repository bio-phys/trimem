import helfrich.core as m
import meshzoo
import numpy as np

import pytest

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
def get_mesh():
    """Get a mesh."""
    points, cells = meshzoo.icosa_sphere(8)
    return m.TriMesh(points, cells)

def get_energy_manager(mesh, bond_type):
    """Setup energy manager."""
    a, l = m.avg_tri_props(mesh)

    bparams = m.BondParams()
    bparams.type = bond_type
    bparams.lc0  = 1.15*l
    bparams.lc1  = 0.85*l
    bparams.a0   = a

    eparams = m.EnergyParams()
    eparams.kappa_b        = 1.0
    eparams.kappa_a        = 1.0
    eparams.kappa_v        = 1.0
    eparams.kappa_c        = 1.0
    eparams.kappa_t        = 1.0
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
    return request.param

# -----------------------------------------------------------------------------
#                                                                      tests --
# -----------------------------------------------------------------------------
def test_gradient(bond_type):
    """Test gradient."""

    mesh   = get_mesh()
    estore = get_energy_manager(mesh, bond_type)

    # finite difference gradient
    ref_grad = np.empty((mesh.n_vertices(),3))
    m.gradient(mesh, estore, ref_grad, 1.0e-8)

    # analytic gradient
    grad = estore.gradient(mesh)

    assert np.linalg.norm(grad - ref_grad)/np.linalg.norm(ref_grad) < 1.0e-4
