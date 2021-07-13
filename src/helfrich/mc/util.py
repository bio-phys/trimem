import numpy as np
from .. import _core as m
from .. import openmesh as om

CONF = """[DEFAULT]
num_steps = 1000
info = 1
[BONDS]
bond_type = Edge
[ENERGY]
kappa_b = 1.0
kappa_a = 1.0
kappa_v = 1.0
kappa_c = 1.0
kappa_t = 1.0
area_fraction = 1.0
volume_fracion = 1.0
curvature_fracion = 1.0
[PARAMETER CONTINUATION]
delta = 1.0
lambda = 1.0
[HMC]
num_steps = 10
step_size = 1.0
"""

def exp_cooling(cfac):
    """exponential cooling schedule."""
    def _cool(step):
        return np.exp(-cfac*step)
    return _cool

def om_helfrich_energy(mesh, estore, istep=10):

    x0 = mesh.points().copy()

    def _fun(x,T):
        points = mesh.points()
        np.copyto(points, x)
        vr = v.ravel()
        e = estore.energy()/T
        np.copyto(points,x0)
        return e

    def _grad(x,T):
        points = mesh.points()
        np.copyto(points,x)
        g = estore.gradient()
        np.copyto(points,x0)
        return g/T

    def _callback(i,acc):
        if i%istep == 0:
            print("\n-- Step ",i)
            print("  ----- acc-rate:   ", acc)
            p = mesh.points()
            np.copyto(p,x)
            estore.energy()
            estore.print_info()
        estore.update_reference_properties()

    return _fun, _grad, _callback

def default_config(fname):
    with open(fname, "w") as fp:
        fp.write(CONF)

def setup_energy_manager(mesh, config):
    """Setup energy manager."""
    l = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    a = m.area(mesh)/mesh.n_faces();

    bparams = m.BondParams()
    bparams.type = config["bond_type"]
    bparams.lc0  = 1.15*l
    bparams.lc1  = 0.85*l
    bparams.a0   = a

    ec = config["energy"]
    eparams = m.EnergyParams()
    eparams.kappa_b        = ec.getfloat("kappa_b")
    eparams.kappa_a        = ec.getfloat("kappa_a")
    eparams.kappa_v        = ec.getfloat("kappa_v")
    eparams.kappa_c        = ec.getfloat("kappa_c")
    eparams.kappa_t        = ec.getfloat("kappa_t")
    eparams.bond_params    = bparams

    cc = config["reference parameter continuation"]
    cparams = m.ContinuationParams()
    cparams.area_frac      = ec.getfloat("area_fraction")
    cparams.volume_frac    = ec.getfloat("volume_fraction")
    cparams.curvature_frac = ec.getfloat("curvature_fraction")
    cparams.delta          = cc.getfloat("delta")
    cparams.lam            = cc.getfloat("lambda")

    return m.EnergyManager(mesh, eparams, cparams)

def write_results(x, mesh, every=1):
    """Write 'trajectories' to files."""

    for i,xi in enumerate(xn):
        if i%every == 0:
          x = mesh.points()
          np.copyto(x,xi)
          om.write_mesh("out/test_"+str(i)+".stl", mesh, binary=True)

def run_hmc(mesh, config):
    """Run hmc on mesh with config."""

    # setup energy manager
    estore = setup_energy_manager(mesh, config)

    # get algorithm parameters
    istep = config.getfloat("info")
    N     = config.getfloat("num_steps")
    cT    = config.getfloat("cooling_factor")
    m     = config["HMC"].getfloat("momentum_variance")
    dt    = config["HMC"].getfloat("step_size")
    L     = config["HMC"].getfloat("num_steps")

    # init callables
    cooling = exp_cooling(cT)
    funs    = om_helfrich_energy(mesh, estore, istep)

    # run hmc
    x0  = mesh.points()
    res = hmc(x0, funcs[0], funcs[1], m, N, dt, L, cooling, funcs[2])

    # write output
    iout = config.getfloat("output_interval")
    write_results(res, mesh, iout)
