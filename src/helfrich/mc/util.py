import warnings

import numpy as np
from scipy.optimize import minimize

from .. import _core as m
from .hmc import MeshHMC, MeshFlips, MeshMonteCarlo
from .mesh import Mesh, read_trimesh
from .config import update_config_defaults, config_to_params
from .output import make_output, CheckpointWriter, CheckpointReader
from .evaluators import EnergyEvaluators


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
      run_mc(mesh, estore, config, restart)
    elif algo == "minimize":
      run_minim(mesh, estore, config, restart)
    else:
      raise ValueError("Invalid algorithm")

def run_mc(mesh, estore, config, restart):
    """Run monte carlo sampling."""

    # construct output writer
    output = make_output(config)

    # function, gradient and callback
    options = {
        "info_step":    config["GENERAL"].getint("info"),
        "output_step":  config["HMC"].getint("thin"),
        "refresh_step": config["SURFACEREPULSION"].getint("refresh"),
    }
    funcs = EnergyEvaluators(mesh, estore, output, options)

    # setup hmc to sample vertex positions
    cmc  = config["HMC"]
    options = {
        "mass":                  cmc.getfloat("momentum_variance"),
        "time_step":             cmc.getfloat("step_size"),
        "num_integration_steps": cmc.getint("traj_steps"),
        "initial_temperature":   cmc.getfloat("initial_temperature"),
        "cooling_factor":        cmc.getfloat("cooling_factor"),
        "cooling_start_step":    cmc.getint("start_cooling"),
        "info_step":             config["GENERAL"].getint("info"),
    }
    hmc = MeshHMC(mesh, funcs.fun, funcs.grad, options=options)

    # setup edge flips
    options = {
        "flip_type":  cmc["flip_type"],
        "flip_ratio": cmc.getfloat("flip_ratio"),
        "info_step":  config["GENERAL"].getint("info"),
    }
    flips = MeshFlips(mesh, estore, options)

    # setup combined-step markov chain
    mmc = MeshMonteCarlo(hmc, flips, callback=funcs.callback)
   
    # run sampling 
    mmc.run(cmc.getint("num_steps"))

    # write checkpoint
    mesh.x = hmc.x
    ec = config["ENERGY"]
    ec["continuation_lambda"] = str(estore.eparams.continuation_params.lam)
    ec["continuation_delta"]  = str(estore.eparams.continuation_params.delta)
    write_checkpoint(mesh, config)

def run_minim(mesh, estore, config, restart):
    """Run minimization."""

    # construct output writer
    output = make_output(config)

    # generic minimization requires surface repulsion refresh at every step
    refresh = config["SURFACEREPULSION"].getint("refresh")
    if not refresh == 1:
        wstr = f"SURFACEREPULSION::refresh is set to {refresh}, " + \
               "which is ignored in in minimization."
        warnings.warn(wstr)
        refresh = 1

    # function, gradient and callback
    options = {
        "info_step":    config["GENERAL"].getint("info"),
        "output_step":  config["MINIMIZATION"].getint("out_every"),
        "refresh_step": refresh,
        "flatten":      True,
    }
    funcs = EnergyEvaluators(mesh, estore, output, options)

    # run minimization
    options = {
        "maxiter": config["MINIMIZATION"].getint("maxiter"),
        "disp": 0,
    }
    res = minimize(
        funcs.fun,
        mesh.x,
        jac=funcs.grad,
        callback=funcs.callback,
        method="L-BFGS-B",
        options=options
    )
    mesh.x = res.x.reshape(mesh.x.shape)

    # print info
    estore.print_info(mesh.trimesh)

    # write checkpoint
    write_checkpoint(mesh, config)
