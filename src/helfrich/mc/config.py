import configparser
import pathlib
import io

from .. import _core as m

CONF = """[GENERAL]
algorithm = hmc
info = 1
# the io-name/prefix parameters are set automatically from the config file
# name; uncomment for more precise control
#input = inp.stl
#output_prefix = inp
#restart_prefix = inp
output_format = vtu
checkpoint_every = 0
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

_bond_enums = {
    "Edge": m.BondType.Edge,
    "Area": m.BondType.Area
}

def write_default_config(fname):
    """Write default config to file."""
    with open(fname, "w") as fp:
        fp.write(CONF)

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
