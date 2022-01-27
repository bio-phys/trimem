"""Output utilities."""

import pathlib
from .xdmf import XdmfWriter
from .xyz import XyzWriter
from .vtu import VtuWriter

_backup_suffixes = [
    ".xyz",
    ".xmf",
    ".h5",
    ".cpt",
    ".vtu",
]

def create_backup(prefix):
    """Create enumerated backups of file-set with prefixed name."""

    fname = pathlib.Path(prefix)

    # find last backup
    bnum = -1
    for f in fname.parent.glob("#*#"):
        bnum = max(int(f.name.strip("#").split(".")[-1]), bnum)

    # backup
    backups = []
    for suffix in _backup_suffixes:
        for f in fname.parent.glob(f"{fname}*{suffix}"):
            backups.append(f.name)
            bname = f"#{f.name}.{str(bnum+1)}#"
            backup = fname.parent.resolve().joinpath(bname)
            _ = f.rename(backup)

    print("Created backup of:", backups)

def make_output(config):
    """Return output writer object."""

    fmt    = config["GENERAL"]["output_format"]
    prefix = config["GENERAL"]["output_prefix"]

    if fmt == "xyz":
        return XyzWriter(prefix)
    elif fmt == "xdmf":
        return XdmfWriter(prefix)
    elif fmt == "vtu":
        return VtuWriter(prefix)
    else:
        raise ValueError("Invalid output format: {}".format(fmt))
