import numpy as np
import pickle

from scipy.optimize import minimize
import meshio

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
output_format = vtu
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
continuation_delta = 0.0
continuation_lambda = 1.0
[HMC]
num_steps = 10
step_size = 1.0
momentum_variance = 1.0
thin = 10
flip_ratio = 0.1
[MINIMIZATION]
maxiter = 10

"""

def default_config(fname):
    with open(fname, "w") as fp:
        fp.write(CONF)

def om_helfrich_energy(mesh, estore, config):

    istep  = config["DEFAULT"].getint("info")
    N      = config["DEFAULT"].getint("num_steps")
    fr     = config["HMC"].getfloat("flip_ratio")
    prefix = config["DEFAULT"]["output_prefix"]
    thin   = config["HMC"].getint("thin")

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

    def _flip(x):
        points = mesh.points()
        np.copyto(points, x)
        flips = m.flip(mesh, estore, fr)
        np.copyto(points, x0)
        return flips

    def _callback(x, i, acc):
        _callback.acc += acc
        flips = _flip(x)
        if i%istep == 0:
            print("\n-- Step ",i)
            print("----- acc-rate: ", _callback.acc/istep)
            print("----- acc-flips:", flips/mesh.n_edges())
            p = mesh.points()
            np.copyto(p, x)
            estore.energy()
            estore.print_info()
            np.copyto(p, x0)
            _callback.acc = 0
        if i%thin == 0:
            write_output(x, mesh, _callback.count, config)
            _callback.count += 1
        estore.update_reference_properties()

    # init callback
    _callback.acc   = 0
    _callback.count = 0

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

    # continuation params might come from restarts instead as from input
    if cparams is None:
        cp = m.ContinuationParams()
        cp.delta = ec.getfloat("continuation_delta")
        cp.lam   = ec.getfloat("continuation_lambda")
    else:
        cp = cparams
    eparams.continuation_params = cp

    estore = m.EnergyManager(mesh, eparams)

    return estore, mesh

def write_output(x, mesh, step, config):
    """Write trajectory output."""

    fmt    = config["DEFAULT"]["output_format"]
    prefix = config["DEFAULT"]["output_prefix"]
    if fmt == "vtu":
        fn = prefix + str(step) + ".vtu"
        meshio.write_points_cells(fn, x, [("triangle", mesh.fv_indices())])
    elif fmt == "xyz":
        fn = prefix + ".xyz"
        rw = "w" if step == 0 else "a"
        with open(fn, rw) as fp:
            np.savetxt(fp,
                       x,
                       fmt=["C\t%.6f", "%.6f", "%.6f"],
                       header="{}\n#".format(mesh.n_vertices()),
                       comments="",
                       delimiter="\t")
    else:
        raise ValueError("Unknown file format for output.")

def write_restart(x, mesh, estore, step, config):
    """Write restart checkpoint."""

    prefix = config["DEFAULT"]["restart_prefix"]

    # write params
    with open(prefix + str(step) + "_params_.cpt", "wb") as fp:
        pickle.dump(estore.eparams.continuation_params, fp)

    # write mesh
    fn = prefix + str(step) + "_mesh_.vtu"
    meshio.write_points_cells(fn, x, [("triangle", mesh.fv_indices())])

def read_restart(restart, config):
    """Read energy manager and mesh from restart."""

    prefix = config["DEFAULT"]["restart_prefix"]

    # read params
    with open(prefix+str(restart)+"_params_.cpt", "rb") as fp:
        cparams = pickle.load(fp)

    # read mesh
    fn = prefix + str(restart) + "_mesh_.vtu"
    mesh = meshio.read(fn)

    return om.TriMesh(mesh.points, mesh.cells[0].data), cparams

def run(config, restart=-1):
    """Run algorithm."""

    # create energy manager and mesh
    if restart == -1:
        estore, mesh = setup_energy_manager(config)
    else:
        mesh, cparams = read_restart(restart, config)
        estore, o_mesh = setup_energy_manager(config, cparams)
        estore.set_mesh(mesh)

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

    # write restart
    write_restart(x, mesh, estore, restart+1, config)

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

    # minimization options
    mopt = {"maxiter": N, "disp": 0}

    # run minimization
    res = minimize(sfun, x0.ravel(), jac=sgrad, method="L-BFGS-B", options=mopt)
    x   = res.x.reshape(x0.shape)
    print(res.message)

    # write output
    write_output(x, mesh, 0, config)

    # write restart
    write_restart(x, mesh, estore, restart+1, config)
