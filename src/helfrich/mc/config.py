import configparser
import pathlib
import io
import os

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

# target surface area fraction wrt. the initial geometry
area_fraction = 1.0

# target volume fraction wrt. the initial geometry
volume_fraction = 1.0

# target curvature fraction wrt. the initial geometry
curvature_fraction = 1.0

# time step for the parameter continuation (choose from: [0,1])
continuation_delta = 0.0

# start time for the parameter continuation (choose from: [0,1])
# should be consistent with initial geometry and the chosen
# area-/volume-/curvature-fraction
continuation_lambda = 1.0

[HMC]
# number of steps in the markov chain
num_steps = 10

# inital step number (default: 0)
# controls start of simulated annealing in combination with 'start_cooling'
# uncomment and set to desired value in case; useful for restarting from
# already cooled states
;init_step = 0

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
    """Write default config to fname."""

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
    """Read config from file."""
    cfile = pathlib.Path(fname)

    if not cfile.exists():
        raise FileNotFoundError(cfile)

    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(fname)

    # set config defaults
    update_config_defaults(
        config,
        init_step=0,
        input=f"{cfile.with_suffix('.stl')}",
        output_prefix=f"{cfile.with_suffix('')}",
        restart_prefix=f"{cfile.with_suffix('')}",
        checkpoint_every=0,
    )

    return config

def print_config(config):
    """Print config to stdout."""

    # print config to string
    with io.StringIO() as sout:
        config.write(sout)
        conf = sout.getvalue().replace("\n\n", "\n")

    info  = "\n------------------\n"
    info += "Run configuration:\n"
    info += conf

    print(info)

def update_config_defaults(config, **kwargs):
    """Update default section with kwargs."""
    config.read_dict({"DEFAULT": kwargs})

def config_to_params(config):
    """Translate config to energy params."""

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

    # translate energy params
    ec = config["ENERGY"]
    cp = m.ContinuationParams()
    cp.delta = ec.getfloat("continuation_delta")
    cp.lam   = ec.getfloat("continuation_lambda")

    eparams = m.EnergyParams()
    eparams.kappa_b             = ec.getfloat("kappa_b")
    eparams.kappa_a             = ec.getfloat("kappa_a")
    eparams.kappa_v             = ec.getfloat("kappa_v")
    eparams.kappa_c             = ec.getfloat("kappa_c")
    eparams.kappa_t             = ec.getfloat("kappa_t")
    eparams.kappa_r             = ec.getfloat("kappa_r")
    eparams.area_frac           = ec.getfloat("area_fraction")
    eparams.volume_frac         = ec.getfloat("volume_fraction")
    eparams.curvature_frac      = ec.getfloat("curvature_fraction")
    eparams.bond_params         = bparams
    eparams.repulse_params      = rparams
    eparams.continuation_params = cp

    return eparams
