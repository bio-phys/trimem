"""Trimem run module.

High level building blocks to run simulations. These blocks are utilized
from the `mc_app` cli but can also be used standalone as a python module.
"""

import warnings
import functools
import copy

import numpy as np
from scipy.optimize import minimize

from .. import core as m
from .hmc import MeshHMC, MeshFlips, MeshMonteCarlo
from .mesh import Mesh, read_trimesh
from .config import update_config_defaults, config_to_params, print_config
from .output import make_output, create_backup, \
                    CheckpointWriter, CheckpointReader
from .evaluators import TimingEnergyEvaluators
from .. import __version__


def setup_energy_manager(config):
    """Setup energy manager.

    Create EnergyManager and Mesh from config file.

    Args:
        config (dict-like): run-config file.

    Returns:
        A tuple (estore, mesh) where estore is of type :class:`EnergyManager`
        and mesh is of type :class:`Mesh`.
    """

    mesh = read_trimesh(config["GENERAL"]["input"])

    # reference values for edge_length and face_area
    a, l = m.avg_tri_props(mesh.trimesh)

    update_config_defaults(config, lc0=1.25*l, lc1=0.75*l, a0=a)
    eparams = config_to_params(config)

    estore = m.EnergyManager(mesh.trimesh, eparams)

    return estore, mesh

def write_checkpoint_handle(config, fix_step=None):
    """Create checkpoint write handle.

    Args:
        config (dict-like): run-config file.

    Keyword Args:
        fix_step (None or int): fix step input in handle signature.

    Returns:
        A function handle with signature (mesh, estore, step) that allows to
        write the mesh and the state of the EnergyManager to a checkpoint
        file. 'step' can be fixed to a particular value by the `fix_step`
        argument.
    """

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
    """Read checkpoint file.

    Acquire checkpoint prefix from config and read the checkpoint with
    number 'restartnum'.

    Note: Continuation params are taken from the checkpoint.
    Note: 'init_step' is set from the HMC section if given.

    Args:
        config (dict-like): run-config file.
        restartnum (int): checkpoint file number to read.

    Returns:
        A tuple (mesh, config) with mesh being of type :class:`Mesh` and
        config being of type `ConfigParser`.
    """

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

    print("Read checkpoint:", cpt.fname)

    return Mesh(points, cells), config

def run(config, restart=None):
    """Run algorithm.

    Runs an algorithm defined by the run-config. Performs a restart in case.

    Args:
        config (dict-like): run-config file.
        restart (None or int): checkpoint file number to restart from.
    """

    # print start info
    print("Running with trimem version {}".format(__version__))

    # do backup for non-restarts
    if restart is None:
        create_backup(
            config["GENERAL"]["output_prefix"],
            config["GENERAL"]["restart_prefix"]
        )

    # setup mesh and energy
    if restart is None:
        estore, mesh = setup_energy_manager(config)
    else:
        mesh, config = read_checkpoint(config, restart)
        estore, _    = setup_energy_manager(config)
        estore.update_repulsion(mesh.trimesh)

    # print effective run configuration
    print_config(config)

    # run algorithm
    algo    = config["GENERAL"]["algorithm"]
    if algo == "hmc":
      run_mc(mesh, estore, config)
    elif algo == "minimize":
      run_minim(mesh, estore, config)
    else:
      raise ValueError("Invalid algorithm")

def run_mc(mesh, estore, config):
    """Run Monte Carlo sampling.

    Perform Monte Carlo sampling of the Helfrich bending energy as defined
    by the `config`.

    Args:
        mesh (mesh.Mesh): initial geometry.
        estore (EnergyManager): EnergyManager.
        config (dict-like): run-config file.
    """

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

def run_minim(mesh, estore, config):
    """Run (precursor) minimization.

    Performs a minimization of the Helfrich bending energy as defined
    by the `config`.

    Args:
        mesh (mesh.Mesh): initial geometry.
        estore (EnergyManager): EnergyManager.
        config (dict-like): run-config file.
    """

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
