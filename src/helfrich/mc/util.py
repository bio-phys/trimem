import warnings

import numpy as np
import pickle

from scipy.optimize import minimize
import meshio

from .. import _core as m
from .hmc import hmc
from .mesh import Mesh, read_trimesh

def om_helfrich_energy(mesh, estore, config):

    istep  = config["GENERAL"].getint("info")
    N      = config["GENERAL"].getint("num_steps")
    fr     = config["HMC"].getfloat("flip_ratio")
    ft     = config["HMC"]["flip_type"]
    prefix = config["GENERAL"]["output_prefix"]
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

    def _fun(x):
        mesh.x = x
        e = estore.energy(mesh.trimesh)
        return e

    def _grad(x):
        mesh.x = x
        g = estore.gradient(mesh.trimesh)
        return g

    def _callback(x, i, acc, T):
        _callback.acc += acc
        mesh.x = x
        flips = _tflips(mesh.trimesh, estore, fr)
        if i%istep == 0:
            print("\n-- Step ",i)
            print("----- acc-rate:   ", _callback.acc/istep)
            print("----- acc-flips:  ", flips/mesh.trimesh.n_edges())
            print("----- temperature:", T)
            estore.print_info(mesh.trimesh)
            _callback.acc = 0
        if i%thin == 0:
            write_output(mesh, _callback.count, config)
            _callback.count += 1
        if i%rfrsh == 0:
            estore.update_repulsion(mesh.trimesh)
        estore.update_reference_properties()

    # init callback
    _callback.acc   = 0
    _callback.count = 0

    return _fun, _grad, _callback

def setup_energy_manager(config, cparams=None):
    """Setup energy manager."""

    mesh = read_trimesh(config["GENERAL"]["input"])

    # reference values for edge_length and face_area
    l = np.mean([mesh.trimesh.calc_edge_length(he)
                 for he in mesh.trimesh.halfedges()])
    a = m.area(mesh.trimesh)/mesh.trimesh.n_faces();

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

    estore = m.EnergyManager(mesh.trimesh, eparams)

    return estore, mesh

def write_output(mesh, step, config):
    """Write trajectory output."""

    fmt    = config["GENERAL"]["output_format"]
    prefix = config["GENERAL"]["output_prefix"]
    if fmt == "vtu":
        fn = prefix + str(step) + ".vtu"
        meshio.write_points_cells(fn, mesh.x, [("triangle", mesh.f)])
    elif fmt == "xyz":
        fn = prefix + ".xyz"
        rw = "w" if step == 0 else "a"
        with open(fn, rw) as fp:
            np.savetxt(fp,
                       mesh.x,
                       fmt=["C\t%.6f", "%.6f", "%.6f"],
                       header="{}\n#".format(len(mesh.x)),
                       comments="",
                       delimiter="\t")
    else:
        raise ValueError("Unknown file format for output.")

def write_restart(mesh, estore, step, config):
    """Write restart checkpoint."""

    prefix = config["GENERAL"]["restart_prefix"]

    # write params
    with open(prefix + str(step) + "_params_.cpt", "wb") as fp:
        pickle.dump(estore.eparams.continuation_params, fp)

    # write mesh
    fn = prefix + str(step) + "_mesh_.vtu"
    meshio.write_points_cells(fn, mesh.x, [("triangle", mesh.f)])

def read_restart(restart, config):
    """Read energy manager and mesh from restart."""

    prefix = config["GENERAL"]["restart_prefix"]

    # read params
    with open(prefix+str(restart)+"_params_.cpt", "rb") as fp:
        cparams = pickle.load(fp)

    # read mesh
    fn = prefix + str(restart) + "_mesh_.vtu"
    mesh = meshio.read(fn)

    return Mesh(mesh.points, mesh.cells[0].data), cparams

def run(config, restart=-1):
    """Run algorithm."""

    # create energy manager and mesh
    if restart == -1:
        estore, mesh = setup_energy_manager(config)
    else:
        mesh, cparams = read_restart(restart, config)
        estore, _ = setup_energy_manager(config, cparams)

    # run algorithm
    algo    = config["GENERAL"]["algorithm"]
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
    x, traj = hmc(mesh.x, fun, grad, options)

    # write restart
    mesh.x = x
    write_restart(mesh, estore, restart+1, config)

def run_minim(mesh, estore, config, restart):
    """Run minimization."""

    # parameters
    x0   = mesh.x.copy()
    N    = config["MINIMIZATION"].getint("maxiter")
    outi = config["MINIMIZATION"].getint("out_every")
    info = config["GENERAL"].getint("info")

    # generic minimization requires surface repulsion refresh at every step
    refresh = config["SURFACEREPULSION"].getint("refresh")
    if not (refresh == 1 or refresh is None):
        wstr = "SURFACEREPULSION::refresh is set to {}, ".format(refresh) + \
               "which is ignored in in minimization."
        warnings.warn(wstr)

    # generate energy, gradient and callback callables to scipy interface
    def _fun(x):
        mesh.x = x.reshape(x0.shape)
        estore.update_repulsion(mesh.trimesh)
        return estore.energy(mesh.trimesh)

    def _grad(x):
        mesh.x = x.reshape(x0.shape)
        estore.update_repulsion(mesh.trimesh)
        return estore.gradient(mesh.trimesh).ravel()

    def _cb(x, force_info=False):
        mesh.x = x.reshape(x0.shape)
        if force_info or (info > 0 and _cb.i % info == 0):
            print("\niter:",_cb.i)
            estore.print_info(mesh.trimesh)
        if outi > 0 and _cb.i % outi == 0:
            write_output(mesh, _cb.count, config)
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

    # print info (also updates the mesh)
    _cb(x, force_info=True)

    # write output
    write_output(mesh, _cb.count, config)

    # write restart
    write_restart(mesh, estore, restart+1, config)
