import numpy as np
import pickle
from scipy.optimize import minimize

from .. import _core as m
from .. import openmesh as om
from .hmc import hmc

CONF = """[DEFAULT]
algorithm = hmc
num_steps = 1000
info = 1
input = "test.stl"
output_prefix = out/test_
restart_prefix = out/restart_
[BONDS]
bond_type = Edge
[ENERGY]
kappa_b = 1.0
kappa_a = 1.0
kappa_v = 1.0
kappa_c = 1.0
kappa_t = 1.0
area_fraction = 1.0
volume_fraction = 1.0
curvature_fraction = 1.0
[HMC]
num_steps = 10
step_size = 1.0
momentum_variance = 1.0
thin = 10
[MINIMIZATION]
maxiter = 10

"""

def default_config(fname):
    with open(fname, "w") as fp:
        fp.write(CONF)

def om_helfrich_energy(mesh, estore, config):

    istep  = config["DEFAULT"].getint("info")
    N      = config["DEFAULT"].getint("num_steps")
    prefix = config["DEFAULT"]["restart_prefix"]

    x0 = mesh.points().copy()

    def _fun(x):
        points = mesh.points()
        np.copyto(points, x)
        e = estore.energy()
        np.copyto(points, x0)
        return e

    def _grad(x):
        points = mesh.points()
        np.copyto(points, x)
        g = estore.gradient()
        np.copyto(points, x0)
        return g

    def _callback(x, i, acc):
        _callback.acc += acc
        if i%istep == 0:
            print("\n-- Step ",i)
            print("----- acc-rate:", _callback.acc/istep)
            p = mesh.points()
            np.copyto(p, x)
            estore.energy()
            estore.print_info()
            np.copyto(p, x0)
            _callback.acc = 0
        estore.update_reference_properties()

    _callback.acc = 0

    return _fun, _grad, _callback

def setup_energy_manager(config, cparams=None):
    """Setup energy manager."""

    mesh = om.read_trimesh(config["DEFAULT"]["input"])

    # reference values for edge_length and face_area
    l = np.mean([mesh.calc_edge_length(he) for he in mesh.halfedges()])
    a = m.area(mesh)/mesh.n_faces();

    str_to_enum = {"Edge": m.BondType.Edge, "Area": m.BondType.Area}

    bparams = m.BondParams()
    bparams.type = str_to_enum[config["BONDS"]["bond_type"]]
    bparams.r    = config["BONDS"].getint("r")
    bparams.lc0  = 1.15*l
    bparams.lc1  = 0.85*l
    bparams.a0   = a

    ec = config["ENERGY"]
    eparams = m.EnergyParams()
    eparams.kappa_b        = ec.getfloat("kappa_b")
    eparams.kappa_a        = ec.getfloat("kappa_a")
    eparams.kappa_v        = ec.getfloat("kappa_v")
    eparams.kappa_c        = ec.getfloat("kappa_c")
    eparams.kappa_t        = ec.getfloat("kappa_t")
    eparams.area_frac      = ec.getfloat("area_fraction")
    eparams.volume_frac    = ec.getfloat("volume_fraction")
    eparams.curvature_frac = ec.getfloat("curvature_fraction")
    eparams.bond_params    = bparams

    if cparams is None:
        estore = m.EnergyManager(mesh, eparams)
    else:
        estore = m.EnergyManager(mesh, eparams, cparams)

    return estore, mesh


def write_trajectory(x, mesh, prefix):
    """Write trajectory to files."""

    for i,xi in enumerate(x):
        x = mesh.points()
        np.copyto(x,xi)
        om.write_mesh(prefix+str(i)+".stl", mesh, binary=True)

def write_restart(x, estore, step, prefix):
    """Write restart checkpoint."""

    # write nececessary data from energy manager
    with open(prefix+str(step)+"_.cpt", "wb") as fp:
        pickle.dump(estore.cparams, fp)
        pickle.dump(x, fp)

def read_restart(restart, config):
    """Read energy manager from restart."""
    prefix = config["DEFAULT"]["restart_prefix"]
    with open(prefix+str(restart)+"_.cpt", "rb") as fp:
        cparams = pickle.load(fp)
        x       = pickle.load(fp)

    return x, cparams

def run(config, restart=-1):
    """Run algorithm."""

    # create energy manager and mesh
    if restart == -1:
        estore, mesh = setup_energy_manager(config)
    else:
        x, cparams = read_restart(restart, config)
        estore, mesh = setup_energy_manager(config, cparams)
        p = mesh.points()
        np.copyto(p, x)

    # run algorithm
    algo    = config["DEFAULT"]["algorithm"]
    if algo == "hmc":
      run_hmc(mesh, estore, config, restart)
    elif algo == "minimize":
      run_minim(mesh, estore, config, restart)
    else:
      raise ValueError("Invalid algorithm")

def run_hmc(mesh, estore, config, restart):
    """Run hamiltonian monte carlo."""

    # function, gradient and callback
    fun, grad, cb = om_helfrich_energy(mesh, estore, config)

    # run hmc
    x0   = mesh.points()
    cmc  = config["HMC"]
    N    = cmc.getint("num_steps")
    m    = cmc.getfloat("momentum_variance")
    dt   = cmc.getfloat("step_size")
    L    = cmc.getint("traj_steps")
    thin = cmc.getint("thin")
    x, traj = hmc(x0, fun, grad, m, N, dt, L, cb, thin)

    # write output
    prefix = config["DEFAULT"]["output_prefix"]
    write_trajectory(traj, mesh, prefix)

    # write restart
    prefix = config["DEFAULT"]["restart_prefix"]
    write_restart(x, estore, restart+1, prefix)

def run_minim(mesh, estore, config, restart):
    """Run minimization."""

    # function and gradient
    fun, grad, _ = om_helfrich_energy(mesh, estore, config)

    # parameters
    x0 = mesh.points()
    N  = config["MINIMIZATION"].getint("maxiter")

    # adjust callables to scipy interface
    sfun  = lambda x: fun(x.reshape(x0.shape))
    sgrad = lambda x: grad(x.reshape(x0.shape)).ravel()

    # run minimization
    res = minimize(sfun, x0.ravel(), jac=sgrad, options={"maxiter": N})
    print(res.message)

    # write output
    prefix = config["DEFAULT"]["output_prefix"]
    write_trajectory([res.x.reshape(x0.shape)], mesh, prefix)

    # write restart
    prefix = config["DEFAULT"]["restart_prefix"]
    x = res.x.reshape(x0.shape)
    write_restart(x, estore, restart+1, prefix)
