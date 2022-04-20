"""Lighweight wrapper around OpenMesh::TriMesh.

Gives easy and consistent access to the simulation state, i.e., vertices and
triangles.
"""

import numpy as np
import copy

from .. import core as m

class Mesh:
    """Lightweight wrapper around OpenMesh::TriMesh."""

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
    """Read a mesh from file."""
    mesh = Mesh()
    mesh.trimesh = m.read_mesh(filename)
    return mesh
