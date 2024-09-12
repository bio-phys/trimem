"""Test the sampling routines from trimem.mc"""
import numpy as np
import pytest

from trimem.mc.mesh import Mesh
from trimem.mc.hmc import HMC, MeshHMC, MeshFlips, MeshMonteCarlo

from util import icosahedron


# ------------------------------------------------------------------------------
#                                                               reference data -
# ------------------------------------------------------------------------------
ref_hmc = np.array(
    [[1.21514538361521e+00, 3.45092705490263e-01, 1.72048043207970e-01],
     [8.83094835380889e-01, -5.18493776004969e-01, 2.30680693852256e-01],
     [-3.86588367294820e-01, 6.00906203067165e-02, -1.59236652307017e+00],
     [1.34186707049293e+00, -8.76196336438621e-01, -2.35369808087982e-01],
     [5.45333865814399e-01, 2.43563377252531e-01, -1.72729690825969e+00],
     [-1.68098653699926e-01, 1.16378155638353e+00, 2.48272552907107e-01],
     [-1.98495092993358e-01, 1.17524064241584e+00, -1.71145853340613e-01],
     [1.09029436068965e+00, -3.56030042245463e-02, 2.66805796636113e-01],
     [5.53856082562936e-01, -2.29035389519608e-01, -3.89267327581926e-01],
     [-7.03445578685648e-01, -1.99926479936067e-01, 5.86372454556326e-01],
     [-4.63250219138312e-02, 7.49532913971889e-01, 1.25393606371644e+00],
     [3.47575257909102e-03, 1.32944039374819e+00, -9.90400745855732e-01]]
)

# ------------------------------------------------------------------------------
#                                                                        tests -
# ------------------------------------------------------------------------------
def test_hmc():
    """Test plain HMC."""
    np.random.seed(42)

    p, c = icosahedron()
    mesh = Mesh(p,c)

    # simplified energy and gradient evaluators
    def energy(x):
        return 0.5*x.ravel().dot(x.ravel())

    def gradient(x):
        return x

    opt = {
        "info_step": 10,
        "time_step": 1.0e-1,
    }
    
    hmc = HMC(mesh.x, energy, gradient, options=opt)

    hmc.run(10)

    # generate new reference solution to copy-paste above
    if False:
        import io
        with io.StringIO() as fp:
            np.savetxt(fp, hmc.x, fmt=["[%.14e,", "%.14e,", "%.14e],"])
            print(fp.getvalue())

    assert np.linalg.norm(hmc.x - ref_hmc) < 1.0e-14

def test_mesh_hmc():
    """Test MeshHMC."""

    np.random.seed(42)

    p, c = icosahedron()
    mesh = Mesh(p,c)

    # simplified energy and gradient evaluators, but with mesh access
    def energy(x):
        mesh.x = x
        return 0.5*x.ravel().dot(x.ravel())

    def gradient(x):
        mesh.x = x
        return x

    opt = {
        "info_step": 10,
        "time_step": 1.0e-1,
    }
    
    hmc = MeshHMC(mesh, energy, gradient, options=opt)

    hmc.run(10)

    assert np.allclose(hmc.x,ref_hmc)

@pytest.fixture(params=["flip_parallel", "flip_serial"])
def flip_type(request):
    return request.param.split("_")[1]

@pytest.mark.skip(reason="flips need seeding before this can be tested")
def test_mesh_mc(flip_type):
    """Test full mesh monte carlo, i.e., moves + flips."""
    # TODO: flips need to expose a seeding to make sense for such a test here
    assert True

