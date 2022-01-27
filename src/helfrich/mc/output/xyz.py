"""xyz coordinate writer.

This writes vertex coordinates only, i.e. the mesh is lost. It also writes
data as plain text.
"""

import pathlib
import numpy as np
from ._common import _create_part

class XyzWriter:
    """Write series of point locations in xyz-format."""

    def __init__(self, fname):
        """Init."""
        self.fname = pathlib.Path(fname).with_suffix(".xyz")
        self.fname = _create_part(self.fname)

        self.step_counter = 0

    def write_points_cells(self, points, cells):
        """Write points and ignore cells."""

        rwflag = "w" if self.step_counter == 0 else "a"
        with self.fname.open(mode=rwflag) as fp:
            np.savetxt(fp,
                       points,
                       fmt=["C\t%.6f", "%.6f", "%.6f"],
                       header="{}\n#".format(len(points)),
                       comments="",
                       delimiter="\t")

        self.step_counter += 1
