"""A simple vtu writer.

This writes a mesh as a series of enumerated vtu files that can conveniently
be visualzed as time-series by, e.g., paraview.
"""

import pathlib

from ._common import _create_part, _remsuffix

import meshio


class VtuWriter:
    """Vtu writer to write series of meshes."""

    def __init__(self, fname):
        """Init."""
        self.fname   = pathlib.Path(fname).with_suffix(".0.vtu")
        self.fname   = _create_part(self.fname)

        self.part = self.fname.suffixes[-2]
        self.step = 0

    def write_points_cells(self, points, cells):
        """Write points and cells to vtu series file."""

        fname = _remsuffix(self.fname.name, "".join(self.fname.suffixes))
        fname = self.fname.with_name(f"{fname}.{self.step}{self.part}.vtu")
        meshio.write_points_cells(fname, points, [("triangle", cells)])

        self.step +=1
