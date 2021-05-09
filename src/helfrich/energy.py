import helfrich.openmesh as om
import numpy as np
import meshzoo

def calc_energy(mesh, kappa):
    """Edge based evaluation of the helfrich energy on trimesh."""

    curvature = 0
    surface   = 0
    volume    = 0
    energy    = 0
    for he in mesh.halfedges():
        if not mesh.is_boundary(he):
            fh = mesh.face_handle(he)
            sector_area = mesh.calc_sector_area(he)
            edge_length = mesh.calc_edge_length(he)
            edge_vector = mesh.calc_edge_vector(he)
            edge_angle  = mesh.calc_dihedral_angle(he)
            face_normal = mesh.calc_face_normal(fh)
            face_center = mesh.calc_face_centroid(fh)
            o_he        = mesh.opposite_halfedge_handle(he)
            o_area      = mesh.calc_sector_area(o_he)
            edge_curv   = 0.5 * edge_angle * edge_length

            curvature += edge_curv
            surface   += sector_area
            volume    += np.dot(face_normal, face_center) * sector_area / 3
            energy    += edge_curv**2 / (sector_area + o_area)

    # correct multiplicity
    energy    /= 2
    curvature /= 2
    surface   /= 3
    volume    /= 3
    return 2 * kappa * energy, surface, volume, curvature


def calc_energy_v(mesh, kappa):
    """Vertex based evaluation of the helfrich energy on trimesh."""

    curvature = 0
    surface   = 0
    volume    = 0
    energy    = 0
    for ve in mesh.vertices():
        c = 0
        s = 0
        v = 0
        for he in om.VertexOHalfedgeIter(mesh, ve):
            if not mesh.is_boundary(he):
                fh = mesh.face_handle(he)
                sector_area = mesh.calc_sector_area(he)
                edge_length = mesh.calc_edge_length(he)
                edge_vector = mesh.calc_edge_vector(he)
                edge_angle  = mesh.calc_dihedral_angle(he)
                face_normal = mesh.calc_face_normal(fh)
                face_center = mesh.calc_face_centroid(fh)
                edge_curv   = 0.25 * edge_angle * edge_length

                c += edge_curv
                s += sector_area / 3 # assign only 1/3 of each face's area
                v += np.dot(face_normal, face_center) * sector_area / 3

        surface   += s
        volume    += v
        curvature += c
        energy    += 2 * kappa * c**2/s

    # correct multiplicity
    volume  /= 3
    return energy, surface, volume, curvature
