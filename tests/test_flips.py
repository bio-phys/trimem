import helfrich.core as m
import numpy as np
import meshzoo

import pytest

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
@pytest.fixture()
def params():
    """Energy parameters."""

    p, c= meshzoo.uv_sphere(num_points_per_circle=20, num_circles=10)
    mesh = m.TriMesh(p,c)

    a, l = m.avg_tri_props(mesh)
    params = m.BondParams()
    params.type = m.BondType.Edge
    params.r = 2
    params.lc0 = 1.15*l
    params.lc1 = 0.85*l
    params.a0   = a

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

    class Data:
        pass

    data = Data()

    data.estore = estore
    data.mesh = mesh

    yield data

# -----------------------------------------------------------------------------
#                                                                       test --
# -----------------------------------------------------------------------------
def test_flips(params):
    """Test flipping of edges."""

    estore = params.estore
    mesh   = params.mesh

    acc = m.flip(mesh, estore, 0.5)

    assert acc > 10
