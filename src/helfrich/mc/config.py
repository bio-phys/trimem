import configparser

CONF = """[GENERAL]
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

def write_default_config(fname):
    """Write default config to file."""
    with open(fname, "w") as fp:
        fp.write(CONF)

def read_config(fname):
    """Read config from file."""
    config = configparser.ConfigParser(inline_comment_prefixes=("#", ";"))
    config.read(fname)
    return config
