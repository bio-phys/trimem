import io
import subprocess
import configparser

import meshzoo
import meshio

import pytest

CONF = """[GENERAL]
algorithm = {algorithm}
info = 1
input = {stlfile!s}
output_prefix = {outprefix!s}
restart_prefix = {restartprefix!s}
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

# -----------------------------------------------------------------------------
#                                                                     pytest --
# -----------------------------------------------------------------------------
@pytest.fixture(scope="session")
def iodir(tmp_path_factory):
    """Generate input."""

    # prepare io folder
    tmpdir = tmp_path_factory.mktemp("testio")
    tmpdir.joinpath("out").mkdir()

    # define files and prefixes
    stlfile       = tmpdir.joinpath("inp.stl")
    outprefix     = tmpdir.joinpath("out/test_")
    restartprefix = tmpdir.joinpath("out/restart_")
    inconf_hmc    = tmpdir.joinpath("conf_hmc.inp")
    inconf_min    = tmpdir.joinpath("conf_min.inp")
   
    # generate input geometry
    p, c = meshzoo.icosa_sphere(4)
    meshio.write_points_cells(str(stlfile.resolve()), p, [("triangle", c)])

    # generate input file for minimzation
    config = CONF.format(algorithm="minimize",
                         stlfile=stlfile.resolve(),
                         outprefix=outprefix.resolve(),
                         restartprefix=restartprefix.resolve())
    inconf_min.write_text(config)

    # generate input file for hmc
    config = CONF.format(algorithm="hmc",
                         stlfile=stlfile.resolve(),
                         outprefix=outprefix.resolve(),
                         restartprefix=restartprefix.resolve())
    inconf_hmc.write_text(config)

    return tmpdir

# -----------------------------------------------------------------------------
#                                                                       test --
# -----------------------------------------------------------------------------
def test_mcapp_config(iodir):
    """Test default config generation."""

    conf_out = str(iodir.joinpath("conf_default.inp").resolve())

    cmd = ["mc_app", "config", "--conf", conf_out]

    r = subprocess.run(cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=False)

    if not r.stderr is None:
        print(r.stderr.decode())

    # check success of config generation
    assert r.returncode == 0

    # read in generated config
    with open(conf_out, "r") as fp:
        config = fp.read()

    # reference config
    ref_config = (CONF.format(algorithm="hmc",
                              stlfile="test.stl",
                              outprefix="out/test_",
                              restartprefix="out/restart_"))

    # compare configs
    # (string comparison here enables readable output in case of failure)
    assert config == ref_config

def test_mcapp_min(iodir):
    """Test mc_app."""

    conf = str(iodir.joinpath("conf_min.inp").resolve())

    cmd = ["mc_app", "run", "--conf", conf]

    r = subprocess.run(cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=False)

    if not r.stderr is None:
        print(r.stderr.decode())

    assert r.returncode == 0

def test_mcapp_hmc(iodir):
    """Test mc_app restart from minimization."""

    conf = str(iodir.joinpath("conf_hmc.inp").resolve())

    cmd = ["mc_app", "run", "--conf", conf, "--restart", "0"]

    r = subprocess.run(cmd,
                       stdout=subprocess.PIPE,
                       stderr=subprocess.PIPE,
                       check=False)

    if not r.stderr is None:
        print(r.stderr.decode())

    assert r.returncode == 0
  
