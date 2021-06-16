"""
Dihedral angle computes a signed angle between normals,
i.e. concave angels are positive, convex angels negative.
"""

import math

import helfrich.openmesh as om
import numpy as np

z = np.sqrt(2)/2 # -z: conave, z: concex
points = np.array([[.0, .0, .0],
                   [1., .0, .0],
                   [0., 1., .0],
                   [.5, .5,  z]])

cells = np.array([[0,1,2],
                  [1,3,2]])

mesh = om.TriMesh(points, cells)
om.write_mesh("test0.stl", mesh)

for he in mesh.halfedges():
    fh = mesh.face_handle(he)
    if fh.is_valid():

        # dihedral angle from openmesh
        edge_angle    = mesh.calc_dihedral_angle(he)
        face_normal   = mesh.calc_face_normal(fh)

        # helper to compute angle as in trimem
        o_he          = mesh.opposite_halfedge_handle(he)
        o_fh          = mesh.face_handle(o_he)
        o_face_normal = mesh.calc_face_normal(o_fh)

        # this is the angle as computed in trimem
        edge_angle_trimem = math.acos(max(-1., min(np.dot(face_normal, o_face_normal), 1.0)))

        print("--")
        print("EDGE:", mesh.calc_edge_vector(he))
        print(" Normals :",face_normal, o_face_normal)
        print(" Dihedral: (openmesh) {}, (trimem) {}".format(
              edge_angle, edge_angle_trimem))
