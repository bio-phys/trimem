"""xyz coordinate writer.

This writes vertex coordinates only, i.e. the mesh is lost. It also writes
data as plain text.
"""

import io
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

        rwflag = "w+" if self.step_counter == 0 else "a+"
        with self.fname.open(mode=rwflag) as fp:
            fp.write("{}\n#\n".format(len(points)))
            for row in points:
                fmt = "\t".join(["C\t{: .6f}", "{: .6f}", "{: .6f}\n"])
                fp.write(fmt.format(*row))

        self.step_counter += 1
