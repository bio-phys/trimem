"""Lightweight wrapper for :class:`helfrich._core.TriMesh`.

Gives easy and consistent access to the simulation state that is comprised
of vertices and triangles.
"""

import numpy as np
import copy

from .. import core as m

class Mesh:
    """Lightweight wrapper around :class:`helfrich._core.TriMesh`.

    Setters and getters are provided for the vertices and face-indices that
    encapsulate the necessities coping with the fact that the corresponding
    data structures for vertices and faces are not kept on the python side.

    Keyword Args:
        points (ndarray[float]): (N,3) array of vertex positions with N being
            the number of vertices int the mesh.
        cells (ndarray[int]): (M,3) array of triangle definitions with M being
            the number of triangles in the mesh.
    """

    def __init__(self, points=None, cells=None):
        """Initialize mesh given vertices and triangles."""
        if (not points is None) and (not cells is None):
            self.trimesh = m.TriMesh(points, cells)
        else:
            self.trimesh = m.TriMesh()

    # vertex get and set access
    @property
    def x(self):
        """Mesh vertices."""
        return self.trimesh.points()

    @x.setter
    def x(self, values):
        """Set vertices."""
        np.copyto(self.x, values)

    # face get-only access
    @property
    def f(self):
        """Mesh triangles."""
        return self.trimesh.fv_indices()

    def copy(self):
        """Make a deep copy."""
        return copy.deepcopy(self)

def read_trimesh(filename):
    """Read a mesh from file.

    Args:
        filename (str): name of the file to read the mesh from.
    """
    mesh = Mesh()
    mesh.trimesh = m.read_mesh(filename)
    return mesh
