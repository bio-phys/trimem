"""Output utilities."""
import warnings
import pathlib

from .xdmf import XdmfWriter
from .xyz import XyzWriter
from .vtu import VtuWriter

# register suffixes of file-sets subject to backup
_backup_suffixes = {
    "output": [
        ".xyz",
        ".xmf",
        ".h5",
        ".vtu",
    ],
    "checkpoint": [
        ".cpt",
        ".h5",
    ],
}

def create_backup(outprefix, cptprefix):
    """Create enumerated backups of file-set with prefixed names."""

    outfname = pathlib.Path(outprefix)
    cptfname = pathlib.Path(cptprefix)

    # find last backup of a file set with these prefixes
    bnum_out = -1
    for f in outfname.parent.glob(f"#{outfname.name}*#"):
        bnum_out = max(int(f.name.strip("#").split(".")[-1]), bnum_out)
    bnum_cpt = -1
    for f in cptfname.parent.glob(f"#{cptfname.name}*#"):
        bnum_cpt = max(int(f.name.strip("#").split(".")[-1]), bnum_cpt)

    if not bnum_cpt == bnum_out:
        wstr = f"Inconsistent backups found for output ({bnum_out}) " + \
               f"and checkpoint ({bnum_cpt}). Using {max(bnum_out,bnum_cpt)}."
        warnings.warn(wstr)
    bnum = max(bnum_out, bnum_cpt)

    # backup output
    backups = []
    for fname, ftype in [(outfname, "output"), (cptfname, "checkpoint")]:
        for suffix in _backup_suffixes[ftype]:
            for f in fname.parent.glob(f"{fname.name}*{suffix}"):
                backups.append(f.name)
                bname = f"#{f.name}.{bnum+1}#"
                backup = fname.parent.resolve().joinpath(bname)
                _ = f.rename(backup)

    if len(backups):
        print(f"Created backup #{bnum+1} of:", backups)

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
