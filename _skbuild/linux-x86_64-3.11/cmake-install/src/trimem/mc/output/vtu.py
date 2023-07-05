"""A simple vtu writer.

Writes a mesh as a series of enumerated vtu files that can conveniently
be visualized as time-series by, e.g., paraview.
"""

import pathlib

from ._common import _create_part, _remsuffix

import meshio


class VtuWriter:
    """Vtu writer to write series of meshes.

    Args:
        fname (str, path-like): output file prefix.
    """

    def __init__(self, fname):
        """Init."""
        self.fname   = pathlib.Path(fname).with_suffix(".0.vtu")
        self.fname   = _create_part(self.fname)

        self.part = self.fname.suffixes[-2]
        self.step = 0

    def write_points_cells(self, points, cells):
        """Write points and cells to vtu series file.

        Args:
            points (ndarray[float]): (N,3) array of vertex positions with N
                being the number of vertices.
            cells (ndarray[int]): (M,3) array of face definitions with M
                being the number of faces.
        """

        fname = _remsuffix(self.fname.name, "".join(self.fname.suffixes))
        fname = self.fname.with_name(f"{fname}.{self.step}{self.part}.vtu")
        meshio.write_points_cells(fname, points, [("triangle", cells)])

        self.step +=1
