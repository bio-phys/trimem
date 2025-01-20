"""Trimem run module.

High level building blocks to run simulations. These blocks are utilized
from the `mc_app` cli but can also be used standalone as a python module.
"""

import warnings
import functools
import copy
import json
from collections import Counter
import time
from datetime import datetime

import numpy as np
from scipy.optimize import minimize

from .. import core as m
from .hmc import MeshHMC, MeshMutation, MeshMonteCarlo
from .config import update_config_defaults, config_to_params, print_config
from .output import make_output, create_backup, \
                    CheckpointWriter, CheckpointReader
from .. import __version__


def setup_energy_manager(config):
    """Setup energy manager.

    Create EnergyManager and Mesh from config file.

    Args:
        config (dict-like): run-config file.

    Returns:
        A tuple (estore, mesh) where estore is of type :class:`EnergyManager`
        and mesh is of type :class:`TriMesh`.
    """

    mesh = m.read_mesh(config["GENERAL"]["input"])

    # reference values for edge_length and face_area
    a, l = m.avg_tri_props(mesh)

    update_config_defaults(config, lc0=1.25*l, lc1=0.75*l, a0=a)
    eparams = config_to_params(config)

    estore = m.EnergyManager(mesh, eparams)

    return estore

def write_checkpoint_handle(config):
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

    def _write_checkpoint(estore, step={}):
        """Write checkpoint with signature (mesh, estore, step)."""

        if not isinstance(step, dict):
            raise TypeError("kwarg 'step' must be dict-like")

        # update config
        upd = {
            "ENERGY": {
                "area_fraction": estore.eparams.area_fraction,
                "volume_fraction": estore.eparams.volume_fraction,
                "curvature_fraction": estore.eparams.curvature_fraction,
            },
            "EXTERNALPOTENTIAL": {
                "alpha": estore.eparams.external_params.alpha,
                "sigma": estore.eparams.external_params.sigma,
                "height": estore.eparams.external_params.height,
                "radius": estore.eparams.external_params.radius
            },
            "HMC": {
                "init_step": json.dumps(step),
            },
        }
        conf.read_dict(upd)

        prefix = config["GENERAL"]["restart_prefix"]

        cpt = CheckpointWriter(prefix)
        cpt.write(estore.mesh.x, estore.mesh.fv_indices, conf)

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
        A tuple (mesh, config) with mesh being of type :class:`TriMesh`
        and config being of type `ConfigParser`.
    """

    prefix = config["GENERAL"]["restart_prefix"]

    cpt = CheckpointReader(prefix, restartnum)
    points, cells, conf = cpt.read()

    # TODO: restart logic (see issue 26)
    upd = {
        "ENERGY": {
            "area_fraction": conf["ENERGY"]["area_fraction"],
            "volume_fraction": conf["ENERGY"]["volume_fraction"],
            "curvature_fraction": conf["ENERGY"]["curvature_fraction"],
        },
        "EXTERNALPOTENTIAL": {
            "alpha": conf["EXTERNALPOTENTIAL"]["alpha"],
            "radius": conf["EXTERNALPOTENTIAL"]["radius"],
            "height": conf["EXTERNALPOTENTIAL"]["height"],
            "sigma": conf["EXTERNALPOTENTIAL"]["sigma"],
        },
        "DEFAULT": {
            "init_step": conf["HMC"]["init_step"],
        }
    }
    config.read_dict(upd)

    print("Read checkpoint:", cpt.fname)

    return m.TriMesh(points, cells), config

def callback_handle(
    output,
    info_step=100,
    out_step=1000,
    cpt_step=0,
    refresh_step=10,
    write_cpt=lambda e,s: None,
    num_steps=None,
    ):
    """Make a callback handle for minimization and mc."""

    t0 = time.time()
    def _estimate_speed(i):
        if (num_steps is None) or (i == 0):
            return
        dt = time.time() - t0
        speed  = dt / i
        finish = datetime.fromtimestamp(t0 + dt + speed * (num_steps - i))
        print("\n-- Performance measurements")
        print(f"----- estimated speed: {speed:.3e} s/step")
        print(f"----- estimated end:   {finish}")

    def _callback(estore, steps):
        i = sum(steps.values()) #py3.10: steps.total()
        if info_step and (i % info_step == 0):
            print("\n-- Energy-Evaluation-Step ", i)
            estore.print_info()
            _estimate_speed(i)
        if out_step and (i % out_step == 0):
            output.write_points_cells(
                estore.mesh.x,
                estore.mesh.fv_indices,
            )
        if cpt_step and (i % cpt_step == 0):
            write_cpt(estore, steps)
        if refresh_step and (i % refresh_step == 0):
            estore.update_repulsion()
        estore.update()

    return _callback

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
        estore = setup_energy_manager(config)
    else:
        mesh, config = read_checkpoint(config, restart)
        estore       = setup_energy_manager(config)
        estore.mesh  = mesh
        estore.update_repulsion()

    # print effective run configuration
    print_config(config)

    # run algorithm
    algo    = config["GENERAL"]["algorithm"]
    if algo == "hmc":
      run_mc(estore, config)
    elif algo == "minimize":
      run_minim(estore, config)
    else:
      raise ValueError("Invalid algorithm")

def run_mc(estore, config):
    """Run Monte Carlo sampling.

    Perform Monte Carlo sampling of the Helfrich bending energy as defined
    by the `config`.

    Args:
        estore (:class:`EnergyManager`): EnergyManager.
        config (dict-like): run-config file.
    """

    # construct output writer
    output = make_output(config)

    # initialize checkpoint writer
    cpt_writer = write_checkpoint_handle(config)

    istep = config["GENERAL"].getint("info")

    # callback
    options = {
        "info_step":    config["GENERAL"].getint("info"),
        "out_step":     config["HMC"].getint("thin"),
        "cpt_step":     config["GENERAL"].getint("checkpoint_every"),
        "refresh_step": config["SURFACEREPULSION"].getint("refresh"),
        "num_steps":    config["HMC"].getint("num_steps"),
        "write_cpt":    cpt_writer,
    }
    cb = callback_handle(output, **options)

    # list of single MC step to run on the mesh
    steps = []

    # setup hmc to sample vertex positions
    cmc  = config["HMC"]
    dt   = cmc.getfloat("step_size")
    nts  = cmc.getint("traj_steps")
    if not (dt == 0.0 or nts == 0):
        options = {
            "mass":                  cmc.getfloat("momentum_variance"),
            "time_step":             dt,
            "num_integration_steps": nts,
            "initial_temperature":   cmc.getfloat("initial_temperature"),
            "cooling_factor":        cmc.getfloat("cooling_factor"),
            "cooling_start_step":    cmc.getint("start_cooling"),
            "info_step":             istep,
        }
        options = {k: v for k,v in options.items() if not v is None}
        steps.append(MeshHMC(estore, estore.energy, estore.gradient, **options))

    # setup edge flips
    ft = cmc["flip_type"]
    fr = cmc.getfloat("flip_ratio")
    if not (ft == "none" or fr == 0.0):
        if ft == "serial":
            flip_func = m.flip
        elif ft == "parallel":
            flip_func = m.pflip
        else:
            raise ValueError("Wrong flip-type: {}".format(self.ft))
        steps.append(
            MeshMutation(estore, "flips", flip_func, rate=fr, info_step=istep)
        )

    # initialize counters
    step_count = Counter(json.loads(cmc.get("init_step")))

    # setup combined-step markov chain
    mmc = MeshMonteCarlo(steps, step_count, callback=cb)

    # run sampling
    mmc.run(cmc.getint("num_steps"))

    # write final checkpoint
    cpt_writer(estore, mmc.counter)

def run_minim(estore, config):
    """Run (precursor) minimization.

    Performs a minimization of the Helfrich bending energy as defined
    by the `config`.

    Args:
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
    cpt_writer = write_checkpoint_handle(config)

    # function, gradient and callback
    options = {
        "info_step":    config["GENERAL"].getint("info"),
        "out_step":     config["MINIMIZATION"].getint("out_every"),
        "cpt_step":     config["GENERAL"].getint("checkpoint_every"),
        "refresh_step": refresh,
        "num_steps":    config["MINIMIZATION"].getint("maxiter"),
        "write_cpt":    cpt_writer,
    }
    _cb = callback_handle(output, **options)

    # make function handles for scipy
    step_count = Counter()
    def cb(x):
        estore.mesh.x = x.reshape(estore.mesh.x.shape)
        _cb(estore, step_count)
        step_count["move"] += 1

    def fun(x):
        return estore.energy(x.reshape(estore.mesh.x.shape))

    def grad(x):
        return estore.gradient(x.reshape(estore.mesh.x.shape)).ravel()

    # run minimization
    options = {
        "maxiter": config["MINIMIZATION"].getint("maxiter"),
        "disp": 0,
    }
    res = minimize(
        fun,
        estore.mesh.x.ravel(),
        jac=grad,
        callback=cb,
        method="L-BFGS-B",
        options=options
    )
    estore.mesh.x = res.x.reshape(estore.mesh.x.shape)

    # print info
    print("\n-- Minimization finished at iteration", res.nit)
    print(res.message)
    estore.print_info()

    # write final checkpoint
    cpt_writer(estore)
