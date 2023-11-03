"""Trimem run configuration.

Utilities to set up, read and write configuration files providing
control on the trimem functionality.
"""

import importlib
import warnings
import configparser
import pathlib
import io
import os

import numpy as np

from .. import core as m

CONF = """[GENERAL]
# algorithm to run: 'hmc', 'minimize'
algorithm = hmc

# verbosity flag
# print state information every i'th step
info = 1

# initial geometry (default: '<config-file-prefix>.stl')
;input = inp.stl

# output file prefix (default: config-file-prefix)
;output_prefix = inp

# checkpoint file prefix (default: config-file-prefix)
;restart_prefix = inp

# checkpoint frequency (default: 0; only final checkpoint)
;checkpoint_every = 0

# output format (choose from: vtu, xyz, hdmf)
output_format = vtu

[BONDS]
# type of tether potential (choose from: Edge, Area)
bond_type = Edge

# control steepness of penalty potential (must be an integer >= 1)
r = 2

# onset distance of attractive force for 'Edge'-type
# (default: 1.25 * mean edge length computed from initial geometry)
;lc0 =

# onset distance of repelling force for 'Edge'-type
# (default: 0.75 * mean edge length computed from initial geometry)
;lc1 =

# average triangle area for 'Area'-type
# (default: mean triangle area computed from initial geometry)
;a0 =

[SURFACEREPULSION]
# neighbour list algorithm (choose from: cell-list, verlet-list)
n_search = cell-list

# neighbour list cutoff
rlist = 0.1

# vertex self- and direct neighbourhood exclusion
# (choose from: 0,1,2)
# 0: exclude self
# 1: exclude directly connected vertex neighbourhood
# 2: exclude 2-link connected vertex neighbourhood
# exclusion levels are inclusive, i.e., 0<1<2.
exclusion_level = 2

# refresh neighbour lists every i'th step
# for 'algorithm = minimize' this is set to 1 internally
refresh = 1

# onset distance of repulsion force
lc1 = 0.0

# steepness of repulsion force (must be an integer >= 1)
r = 2

[EXTERNALPOTENTIAL]
# type (choose from: 'none', 'sphere'; default: 'none')
type = none

# lennard-jones epsilon
epsilon = 1.0

# lennard-jones sigma
sigma = 1.0

# sphere potential radius
# see `ENERGY.area_fraction` for parameter continuation
radius = 1.0

[ENERGY]
# Helfrich functional weight
kappa_b = 1.0

# surface area penalty weight
kappa_a = 1.0

# volume penalty weight
kappa_v = 1.0

# surface area difference penalty weight
kappa_c = 1.0

# tether penalty weight
kappa_t = 1.0

# repulsion penalty weight
kappa_r = 1.0

# external potential weight (default: currently 0.0 since untested)
kappa_e = 0.0

# target surface area fraction wrt. the initial geometry
# This parameter can be dynamically evaluated during sampling. In
# addition to a single value (that would set the target value ad-hoc)
# also two other modes of specification are understood by trimem:
# 1. $<expr>$ N delta lambda optional-label
#  'expr'  : mathemacical expression given in between tokens '$<' and
#            '>$' as a function of 'x', such as 'sin(x) + cos(x)'. It
#            is evaluated at position x='lambda'. It must evaluate
#            within the scope of the built-in math module.
#  'N'     : positive integer defining the number of intervals at which
#            the above expression is evaluated
#  'lambda': path variable in [0,1] that is used to evolve the
#            respective parameter from fun(0) -> fun(1). The effective
#            value is computed from linear interpolation based on the
#            discretely sampled 'expr'.
#  'delta' : increment to propagate the state 'lambda' from one Monte-
#            Carlo step to the other.
#  'label' : optional label, when given enables a rough terminal-plot
#            of 'expr' over MC-steps.
# 2. start stop delta lambda
#    This results in a linear interpolation of the effective value of
#    the parameter at a certain MC-step by linear interpolation between
#    'start' and 'stop' based on parameter 'lambda'.
area_fraction = 1.0

# target volume fraction wrt. the initial geometry
# see `area_fraction` for parameter continuation
volume_fraction = 1.0

# target curvature fraction wrt. the initial geometry
# see `area_fraction` for parameter continuation
curvature_fraction = 1.0

# time step for the parameter continuation (choose from: [0,1])
continuation_delta = 0.0

# start time for the parameter continuation (choose from: [0,1])
# should be consistent with initial geometry and the chosen
# area-/volume-/curvature-fraction
continuation_lambda = 1.0

[HMC]
# number of steps to run in the markov chain
num_steps = 10

# inital step number counters (default: {})
# if not empty it must be a stringification of a dict with keys in
# ["move", "flip"] and values giving the step count for the step-type indicated
# by the key, e.g., a value of {"move": 10, "flip": 5} would restart with a
# total step count of 15. An empty dict resets all counters.
# this can be used to control the start of simulated annealing in combination
# with 'start_cooling' uncomment and set to desired value in case; useful for
# restarting from already cooled states
;init_step = {}

# step size for time integration within the HMC-step
step_size = 1.0

# number of steps within the time integration of a HMC-step
traj_steps = 10

# mass matrix magnitude for the HMC integration
momentum_variance = 1.0

# keep every i'th step of the markov chain
thin = 10

# precentage of flips to attempt during a flip sweep (choose from: [0,1])
flip_ratio = 0.1

# flip-sweep implementation (choose from: none, serial, parallel)
flip_type = serial

# temperature (cooled down to zero with simulated annealing)
initial_temperature = 1.0

# exponential cooling factor for simulated annealing (choose >= 0)
# larger values correspond to faster cooling
cooling_factor = 1.0e-4

# start cooling at step i
# start simulated annealing for 'start_cooling > (init_step + i)'
start_cooling = 0

[MINIMIZATION]
# maximum number of iterations in the minimization
maxiter = 10

# keep every i'th iteration in the output
out_every = 0
"""

_bond_enums = {
    "Edge": m.BondType.Edge,
    "Area": m.BondType.Area
}

def write_default_config(fname, strip=True):
    """Write default config to fname.

    Args:
        fname (str, path-like): file name to write to.

    Keyword Args:
        strip (bool): Strip comments from output (default: True).
    """

    if strip:
        config = os.linesep.join([l for l in CONF.splitlines()
                                  if l and not l.startswith("#")])
        config += os.linesep # end with newline
    else:
        config = CONF

    if hasattr(fname, "write"):
        fname.write(config)
    else:
        fp = pathlib.Path(fname)
        fp.write_text(config)

def read_config(fname):
    """Read config from file.

    Args:
        fname (str, path-like): file name to read from.
    """
    cfile = pathlib.Path(fname)

    if not cfile.exists():
        raise FileNotFoundError(cfile)

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(fname)

    # set config defaults
    update_config_defaults(
        config,
        init_step="{}",
        input=f"{cfile.with_suffix('.stl')}",
        output_prefix=f"{cfile.with_suffix('')}",
        restart_prefix=f"{cfile.with_suffix('')}",
        checkpoint_every=0,
    )

    return config

def print_config(config):
    """Print config to stdout.

    Args:
        config (ConfigParser): configuration to print.
    """

    # print config to string
    with io.StringIO() as sout:
        config.write(sout)
        conf = sout.getvalue().replace("\n\n", "\n")

    info  = "\n------------------\n"
    info += "Run configuration:\n"
    info += conf

    print(info)

def update_config_defaults(config, **kwargs):
    """Update default section with kwargs.

    Args:
        config (ConfigParser): config to be updated.

    Keyword Args:
        kwargs (dict): dictionary to be written to the `DEFAULT` section.
    """
    config.read_dict({"DEFAULT": kwargs})

def _parse_continuation_specs(spec):
    """Preprocess continuation tuples."""
    try:
        if spec.startswith("$<"):
            end   = spec.find(">$")
            if end == -1:
                msg = "missing expression token: '>$'"
                raise ValueError(msg)
            expr  = spec[2:end]
            rem   = spec[end+2:].split()
            props = rem[:3]
            if not len(props) == 3:
                msg = f"wrong number of parameters: {len(props)}"
                raise ValueError(msg)
            label = "".join(rem[3:])[:8]
            props = [int(props[0]), float(props[1]), float(props[2])]
            tup   = [expr, *props, label if not label=="" else None]
        else:
            tup = [float(i) for i in spec.split()]
            if not ( len(tup) == 1 or len(tup) == 4 ):
                msg = f"wrong number of parameters: {len(tup)}"
                raise ValueError(msg)
    except Exception as e:
        msg = f"Cannot parse continuation tuple: {spec}"
        raise ValueError(msg) from e
    return tup

def config_to_params(config):
    """Translate config to energy params.

    Args:
        config (ConfigParser): config to be translated

    Returns:
        EnergyParams:
            Instance of :class:`helfrich._core.EnergyParams` parametrized by
            `config`.
    """

    # translate bond params
    bc      = config["BONDS"]
    bparams = m.BondParams()
    bparams.type = _bond_enums[bc["bond_type"]]
    bparams.r    = bc.getint("r")
    bparams.lc0  = bc.getfloat("lc0")
    bparams.lc1  = bc.getfloat("lc1")
    bparams.a0   = bc.getfloat("a0")

    # translate repulsion params
    rc      = config["SURFACEREPULSION"]
    rparams = m.SurfaceRepulsionParams()
    rparams.n_search        = rc["n_search"]
    rparams.rlist           = rc.getfloat("rlist")
    rparams.exclusion_level = rc.getint("exclusion_level")
    rparams.lc1             = rc.getfloat("lc1")
    rparams.r               = rc.getint("r")

    ex = config["EXTERNALPOTENTIAL"]
    exparams = m.ExternalPotentialParams()
    exparams.type    = ex["type"]
    exparams.epsilon = ex.getfloat("epsilon")
    exparams.sigma   = ex.getfloat("sigma")
    exparams.radius  = m.ContinuationTuple(
        *_parse_continuation_specs(ex.get("radius"))
    )

    # translate energy params
    ec = config["ENERGY"]
    eparams = m.EnergyParams()
    eparams.kappa_b             = ec.getfloat("kappa_b")
    eparams.kappa_a             = ec.getfloat("kappa_a")
    eparams.kappa_v             = ec.getfloat("kappa_v")
    eparams.kappa_c             = ec.getfloat("kappa_c")
    eparams.kappa_t             = ec.getfloat("kappa_t")
    eparams.kappa_r             = ec.getfloat("kappa_r")
    eparams.kappa_e             = ec.getfloat("kappa_e")

    eparams.area_frac           = m.ContinuationTuple(
        *_parse_continuation_specs(ec.get("area_fraction"))
    )
    eparams.volume_frac         = m.ContinuationTuple(
        *_parse_continuation_specs(ec.get("volume_fraction"))
    )
    eparams.curvature_frac      = m.ContinuationTuple(
        *_parse_continuation_specs(ec.get("curvature_fraction"))
    )

    eparams.bond_params         = bparams
    eparams.repulse_params      = rparams
    eparams.external_params     = exparams

    return eparams

def termplot(line, label):
    try:
        plt = importlib.import_module('plotille')
    except ModuleNotFoundError as e:
        msg = "module 'plotille' needed for plotting"
        warnings.warn(msg)
    N     = len(line)
    x     = np.arange(N)
    fig = plt.plot(
        x,
        line,
        width=58,
        height=20,
        x_min=0,
        x_max=N,
        origin=False,
        Y_label=label,
        X_label='steps',
    )
    print('\n'+fig)
