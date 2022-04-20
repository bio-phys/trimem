import io
import argparse

from .config import write_default_config, read_config
from .util import run as irun

# helpers
def run(args):
    """Command action to run hmc."""
    config = read_config(args.conf)
    irun(config, args.restart)

def config(args):
    """Command action to write default config."""
    if args.conf:
        write_default_config(args.conf, args.strip)
    else:
        out = io.StringIO()
        write_default_config(out, args.strip)
        print(out.getvalue())

def cli():
    """command line interface."""

    descr = "Monte Carlo Sampling on om-helfrich energy functional"
    parser = argparse.ArgumentParser(description=descr)
    parser.set_defaults(func=lambda x: parser.print_usage())
    subparsers = parser.add_subparsers(title="subcommands")

    # config subparser
    parse_config = subparsers.add_parser("config", help="write default config")
    parse_config.add_argument("--strip",
                              help="strip comments",
                              action="store_true")
    parse_config.add_argument("--conf",
                              help="config-file name (empty or str)",
                              default="",
                              type=str)
    parse_config.set_defaults(func=config)

    # run subparser
    parse_run = subparsers.add_parser("run", help="run simulation")
    parse_run.add_argument("--conf", help="config-file name", required=True)
    parse_run.add_argument("--restart",
                           help="restart index",
                           nargs="?",
                           default=None,
                           const=-1,
                           type=int)
    parse_run.set_defaults(func=run)

    # execute command
    args = parser.parse_args()
    args.func(args)
