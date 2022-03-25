import warnings
import functools
import copy

import numpy as np
from scipy.optimize import minimize

from .. import _core as m
from .hmc import MeshHMC, MeshFlips, MeshMonteCarlo
from .mesh import Mesh, read_trimesh
from .config import update_config_defaults, config_to_params
from .output import make_output, create_backup, \
                    CheckpointWriter, CheckpointReader
from .evaluators import TimingEnergyEvaluators


def setup_energy_manager(config):
    """Setup energy manager."""

    mesh = read_trimesh(config["GENERAL"]["input"])

    # reference values for edge_length and face_area
    a, l = m.avg_tri_props(mesh.trimesh)

    update_config_defaults(config, lc0=1.25*l, lc1=0.75*l, a0=a)
    eparams = config_to_params(config)

    estore = m.EnergyManager(mesh.trimesh, eparams)

    return estore, mesh

def write_checkpoint_handle(config, fix_step=None):
    """Return function to write checkpoint file."""

    conf = copy.deepcopy(config)

    def _write_checkpoint(mesh, estore, step):
        """Write checkpoint with signature (mesh, estore, step)."""

        # don't always set hmc-init-step, e.g. when doing minimzation
        if not fix_step is None:
            step = fix_step

        # update config
        upd = {
            "ENERGY": {
                "continuation_lambda": estore.eparams.continuation_params.lam,
                "continuation_delta":  estore.eparams.continuation_params.delta,
            },
            "HMC": {
                "init_step": config["HMC"].getint("init_step") + step,
            },
        }
        conf.read_dict(upd)

        prefix = config["GENERAL"]["restart_prefix"]

        cpt = CheckpointWriter(prefix)
        cpt.write(mesh.x, mesh.f, conf)

        print("Writing checkpoint:", cpt.fname)

    return _write_checkpoint

def read_checkpoint(config, restartnum):
    """Read data from checkpoint file."""

    prefix = config["GENERAL"]["restart_prefix"]

    cpt = CheckpointReader(prefix, restartnum)
    points, cells, conf = cpt.read()

    # TODO: restart logic (see issue 26)
    upd = {
        "ENERGY": {
            "continuation_delta":  conf["ENERGY"]["continuation_delta"],
            "continuation_lambda": conf["ENERGY"]["continuation_lambda"],
        },
        "DEFAULT": {
            "init_step": conf["HMC"]["init_step"],
        }
    }
    config.read_dict(upd)

    return Mesh(points, cells), config

def run(config, restart=-1):
    """Run algorithm."""

    # do backup for non-restarts
    if restart == -1:
        create_backup(
            config["GENERAL"]["output_prefix"],
            config["GENERAL"]["restart_prefix"]
        )

    # setup mesh and energy
    if restart == -1:
        estore, mesh = setup_energy_manager(config)
    else:
        mesh, config = read_checkpoint(config, restart)
        estore, _    = setup_energy_manager(config)
        estore.update_repulsion(mesh.trimesh)

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

    # initialize checkpoint writer
    cpt_writer = write_checkpoint_handle(config)

    # function, gradient and callback
    options = {
        "info_step":    config["GENERAL"].getint("info"),
        "output_step":  config["HMC"].getint("thin"),
        "cpt_step":     config["GENERAL"].getint("checkpoint_every"),
        "refresh_step": config["SURFACEREPULSION"].getint("refresh"),
        "init_step":    config["HMC"].getint("init_step"),
        "num_steps":    config["HMC"].getint("num_steps"),
        "write_cpt":    cpt_writer,
    }
    funcs = TimingEnergyEvaluators(mesh, estore, output, options)

    # setup hmc to sample vertex positions
    cmc  = config["HMC"]
    options = {
        "mass":                  cmc.getfloat("momentum_variance"),
        "time_step":             cmc.getfloat("step_size"),
        "num_integration_steps": cmc.getint("traj_steps"),
        "initial_temperature":   cmc.getfloat("initial_temperature"),
        "cooling_factor":        cmc.getfloat("cooling_factor"),
        "cooling_start_step":    cmc.getint("start_cooling"),
        "init_step":             cmc.getint("init_step"),
        "info_step":             config["GENERAL"].getint("info"),
    }
    hmc = MeshHMC(mesh, funcs.fun, funcs.grad, options=options)

    # setup edge flips
    options = {
        "flip_type":  cmc["flip_type"],
        "flip_ratio": cmc.getfloat("flip_ratio"),
        "init_step":  cmc.getint("init_step"),
        "info_step":  config["GENERAL"].getint("info"),
    }
    flips = MeshFlips(mesh, estore, options)

    # setup combined-step markov chain
    mmc = MeshMonteCarlo(hmc, flips, callback=funcs.callback)
   
    # run sampling 
    mmc.run(cmc.getint("num_steps"))

    # update mesh
    mesh.x = hmc.x

    # write final checkpoint
    cpt_writer(mesh, estore, cmc.getint("num_steps"))

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

    # init checkpoint writer (no support/need for 'init_step' in minim)
    cpt_writer = write_checkpoint_handle(config, fix_step=0)

    # function, gradient and callback
    options = {
        "info_step":    config["GENERAL"].getint("info"),
        "output_step":  config["MINIMIZATION"].getint("out_every"),
        "cpt_step":     config["GENERAL"].getint("checkpoint_every"),
        "refresh_step": refresh,
        "flatten":      True,
        "num_steps":    config["MINIMIZATION"].getint("maxiter"),
        "write_cpt":    cpt_writer,
    }
    funcs = TimingEnergyEvaluators(mesh, estore, output, options)

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
    print("\n-- Minimization finished at iteration", res.nit)
    print(res.message)
    estore.print_info(mesh.trimesh)

    # write final checkpoint
    cpt_writer(mesh,estore,0)
