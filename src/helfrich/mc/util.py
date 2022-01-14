import warnings

import numpy as np
import pickle

from scipy.optimize import minimize
import meshio

from .. import _core as m
from .. import openmesh as om
from .hmc import hmc

CONF = """[DEFAULT]
algorithm = hmc
info = 1
input = test.stl
output_prefix = out/test_
restart_prefix = out/restart_
output_format = vtu
[BONDS]
bond_type = Edge
r = 2
[SURFACEREPULSION]
n_search = cell-list
rlist = 0.1
exclusion_level = 2
refresh = 1
lc1 = 0.0
r = 2
[ENERGY]
kappa_b = 1.0
kappa_a = 1.0
kappa_v = 1.0
kappa_c = 1.0
kappa_t = 1.0
kappa_r = 1.0
area_fraction = 1.0
volume_fraction = 1.0
curvature_fraction = 1.0
continuation_delta = 0.0
continuation_lambda = 1.0
[HMC]
num_steps = 10
step_size = 1.0
traj_steps = 10
momentum_variance = 1.0
thin = 10
flip_ratio = 0.1
flip_type = serial
initial_temperature = 1.0
cooling_factor = 1.0e-4
start_cooling = 0
[MINIMIZATION]
maxiter = 10
out_every = 0

"""

def default_config(fname):
    with open(fname, "w") as fp:
        fp.write(CONF)

def om_helfrich_energy(mesh, estore, config):

    istep  = config["DEFAULT"].getint("info")
    N      = config["DEFAULT"].getint("num_steps")
    fr     = config["HMC"].getfloat("flip_ratio")
    ft     = config["HMC"]["flip_type"]
    prefix = config["DEFAULT"]["output_prefix"]
    thin   = config["HMC"].getint("thin")
    rfrsh  = config["SURFACEREPULSION"].getint("refresh")


    if ft == "serial":
        _tflips = lambda mh,e,r: m.flip(mh,e,r)
    elif ft == "parallel":
        _tflips = lambda mh,e,r: m.pflip(mh,e,r)
    elif ft == "none":
        _tflips = lambda mh,e,r: 0
    else:
        raise ValueError("Wrong flip-type")

    # take a short-cut in case
    if fr == 0.0:
        _tflips = lambda mh,e,r: 0

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
        flips = _tflips(mesh, estore, fr)
        np.copyto(points, x0)
        return flips

    def _callback(x, i, acc, T):
        _callback.acc += acc
        flips = _flip(x)
        if i%istep == 0:
            print("\n-- Step ",i)
            print("----- acc-rate:   ", _callback.acc/istep)
            print("----- acc-flips:  ", flips/mesh.n_edges())
            print("----- temperature:", T)
            p = mesh.points()
            np.copyto(p, x)
            estore.energy()
            estore.print_info()
            np.copyto(p, x0)
            _callback.acc = 0
        if i%thin == 0:
            write_output(x, mesh, _callback.count, config)
            _callback.count += 1
        if i%rfrsh == 0:
            p = mesh.points()
            np.copyto(p, x)
            estore.update_repulsion()
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
    bparams.lc0  = 1.25*l
    bparams.lc1  = 0.75*l
    bparams.a0   = a

    rparams = m.SurfaceRepulsionParams()
    rparams.n_search        = config["SURFACEREPULSION"]["n_search"]
    rparams.rlist           = config["SURFACEREPULSION"].getfloat("rlist")
    rparams.exclusion_level = config["SURFACEREPULSION"].getint("exclusion_level")
    rparams.lc1             = config["SURFACEREPULSION"].getfloat("lc1")
    rparams.r               = config["SURFACEREPULSION"].getint("r")

    ec = config["ENERGY"]
    eparams = m.EnergyParams()
    eparams.kappa_b        = ec.getfloat("kappa_b")
    eparams.kappa_a        = ec.getfloat("kappa_a")
    eparams.kappa_v        = ec.getfloat("kappa_v")
    eparams.kappa_c        = ec.getfloat("kappa_c")
    eparams.kappa_t        = ec.getfloat("kappa_t")
    eparams.kappa_r        = ec.getfloat("kappa_r")
    eparams.area_frac      = ec.getfloat("area_fraction")
    eparams.volume_frac    = ec.getfloat("volume_fraction")
    eparams.curvature_frac = ec.getfloat("curvature_fraction")
    eparams.bond_params    = bparams
    eparams.repulse_params = rparams

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
    options = { "number_of_steps": cmc.getint("num_steps"),
                "mass": cmc.getfloat("momentum_variance"),
                "time_step": cmc.getfloat("step_size"),
                "num_integration_steps": cmc.getint("traj_steps"),
                "thin": cmc.getint("thin"),
                "initial_temperature": cmc.getfloat("initial_temperature"),
                "cooling_factor": cmc.getfloat("cooling_factor"),
                "cooling_start_step": cmc.getint("start_cooling"),
                "callback": cb,
              }
    x, traj = hmc(x0, fun, grad, options)

    # write restart
    write_restart(x, mesh, estore, restart+1, config)

def run_minim(mesh, estore, config, restart):
    """Run minimization."""

    # parameters
    x0   = mesh.points().copy()
    N    = config["MINIMIZATION"].getint("maxiter")
    outi = config["MINIMIZATION"].getint("out_every")
    info = config["DEFAULT"].getint("info")

    # generic minimization requires surface repulsion refresh at every step
    refresh = config["SURFACEREPULSION"].getint("refresh")
    if not (refresh == 1 or refresh is None):
        wstr = "SURFACEREPULSION::refresh is set to {}, ".format(refresh) + \
               "which is ignored in in minimization."
        warnings.warn(wstr)

    # generate energy, gradient and callback callables to scipy interface
    def _fun(x):
        p = mesh.points()
        np.copyto(p, x.reshape(x0.shape))
        estore.update_repulsion()
        return estore.energy()

    def _grad(x):
        p = mesh.points()
        np.copyto(p, x.reshape(x0.shape))
        estore.update_repulsion()
        return estore.gradient().ravel()

    def _cb(x, force_info=False):
        if force_info or (info > 0 and _cb.i % info == 0):
            print("\niter:",_cb.i)
            p = mesh.points()
            np.copyto(p, x.reshape(x0.shape))
            estore.energy()
            estore.print_info()
        if outi > 0 and _cb.i % outi == 0:
            p = mesh.points()
            np.copyto(p, x.reshape(x0.shape))
            write_output(p, mesh, _cb.count, config)
            _cb.count += 1
        _cb.i += 1
    _cb.i = 0
    _cb.count = 0

    # minimization options
    mopt = {"maxiter": N, "disp": 0}

    # run minimization
    res = minimize(_fun,
                   x0.ravel(),
                   jac=_grad,
                   callback=_cb,
                   method="L-BFGS-B",
                   options=mopt)
    x   = res.x.reshape(x0.shape)
    print(res.message)

    # print info
    _cb(x, force_info=True)

    # write output
    write_output(x, mesh, _cb.count, config)

    # write restart
    write_restart(x, mesh, estore, restart+1, config)
