import warnings

import numpy as np
import pickle

from scipy.optimize import minimize
import meshio

from .. import _core as m
from .hmc import hmc
from .mesh import Mesh, read_trimesh
from .config import update_config_defaults, config_to_params
from .output import make_output, CheckpointWriter, CheckpointReader

def om_helfrich_energy(mesh, estore, config, output):

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
            output.write_points_cells(mesh.x, mesh.f)
        if i%rfrsh == 0:
            estore.update_repulsion(mesh.trimesh)
        estore.update_reference_properties()

    # init callback
    _callback.acc = 0

    return _fun, _grad, _callback

def setup_energy_manager(config):
    """Setup energy manager."""

    mesh = read_trimesh(config["GENERAL"]["input"])

    # reference values for edge_length and face_area
    a, l = m.avg_tri_props(mesh.trimesh)

    update_config_defaults(config, lc0=1.25*l, lc1=0.75*l, a0=a)
    eparams = config_to_params(config)

    estore = m.EnergyManager(mesh.trimesh, eparams)

    return estore, mesh

def write_checkpoint(mesh, config, **kwargs):
    """Write checkpoint file."""

    prefix = config["GENERAL"]["restart_prefix"]

    cpt = CheckpointWriter(prefix)
    cpt.write(mesh.x, mesh.f, config, **kwargs)

    print("Writing checkpoint:", cpt.fname)

def read_checkpoint(config, restartnum):
    """Read data from checkpoint file."""

    prefix = config["GENERAL"]["restart_prefix"]

    cpt = CheckpointReader(prefix, restartnum)
    points, cells, conf = cpt.read()

    # TODO: restart logic (see issue 10, 12)
    ec = config["ENERGY"]
    ec["continuation_delta"]  = conf["ENERGY"]["continuation_delta"]
    ec["continuation_lambda"] = conf["ENERGY"]["continuation_lambda"]

    return Mesh(points, cells), config

def run(config, restart=-1):
    """Run algorithm."""

    # setup mesh and energy
    if not restart == -1:
        mesh, config = read_checkpoint(config, restart)
        estore, _    = setup_energy_manager(config)
    else:
        estore, mesh = setup_energy_manager(config)

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

    # construct output writer
    output = make_output(config)

    # function, gradient and callback
    fun, grad, cb = om_helfrich_energy(mesh, estore, config, output)

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

    # write checkpoint
    mesh.x = x
    ec = config["ENERGY"]
    ec["continuation_lambda"] = str(estore.eparams.continuation_params.lam)
    ec["continuation_delta"]  = str(estore.eparams.continuation_params.delta)
    write_checkpoint(mesh, config)

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

    # construct output writer
    output = make_output(config)

    # generate energy, gradient and callback callables to scipy interface
    def _fun(x):
        mesh.x = x.reshape(x0.shape)
        estore.update_repulsion(mesh.trimesh)
        return estore.energy(mesh.trimesh)

    def _grad(x):
        mesh.x = x.reshape(x0.shape)
        estore.update_repulsion(mesh.trimesh)
        return estore.gradient(mesh.trimesh).ravel()

    def _cb(x, force_info=False, force_out=False):
        mesh.x = x.reshape(x0.shape)
        if force_info or (info > 0 and _cb.i % info == 0):
            print("\niter:",_cb.i)
            estore.print_info(mesh.trimesh)
        if force_out or (outi > 0 and _cb.i % outi == 0):
            output.write_points_cells(mesh.x, mesh.f)
        _cb.i += 1
    _cb.i = 0

    # minimization options
    mopt = {"maxiter": N, "disp": 0}

    # run minimization
    res = minimize(_fun,
                   x0.ravel(),
                   jac=_grad,
                   callback=_cb,
                   method="L-BFGS-B",
                   options=mopt)
    x   = res.x
    print(res.message)

    # print info (also updates the mesh)
    _cb(x, force_info=True, force_out=True)

    # write checkpoint
    write_checkpoint(mesh, config)
