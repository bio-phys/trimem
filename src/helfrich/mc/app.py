import argparse

from ..openmesh import read_trimesh
from .config import write_default_config, read_config
from .util import run as irun

# helpers
def run(args):
    """Command action to run hmc."""
    config = read_config(args.conf)
    irun(config, args.restart)

def config(args):
    """Command action to write default config."""
    write_default_config(args.conf)

def cli():
    """command line interface."""

    descr = "Monte Carlo Sampling on om-helfrich energy functional"
    parser = argparse.ArgumentParser(description=descr)
    subparsers = parser.add_subparsers(title="subcommands")

    # config subparser
    parse_config = subparsers.add_parser("config", help="write default config")
    parse_config.add_argument("--conf", help="config-file name", required=True)
    parse_config.set_defaults(func=config)

    # run subparser
    parse_run = subparsers.add_parser("run", help="run hmc")
    parse_run.add_argument("--conf", help="config-file name", required=True)
    parse_run.add_argument("--restart", help="restart index", default=-1, type=int)
    parse_run.set_defaults(func=run)

    # execute command
    args = parser.parse_args()
    args.func(args)
