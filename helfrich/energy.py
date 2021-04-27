import helfrich.openmesh as om
import numpy as np
import meshzoo

def calc_energy(mesh, kappa):
    """Evaluate helfrich energy on trimesh."""

    curvature = 0
    surface   = 0
    volume    = 0
    energy    = 0
    for he in mesh.halfedges():
        fh = mesh.face_handle(he)
        if fh.is_valid():
            sector_area = mesh.calc_sector_area(he)
            edge_length = mesh.calc_edge_length(he)
            edge_vector = mesh.calc_edge_vector(he)
            edge_angle  = mesh.calc_dihedral_angle(he)
            face_normal = mesh.calc_face_normal(fh)
            face_center = mesh.calc_face_centroid(fh)
            edge_curv   = 0.5 * edge_angle * edge_length

            curvature += edge_curv
            surface   += sector_area
            volume    += np.dot(face_normal, face_center) * sector_area / 3
            energy    += edge_curv**2

    # correct multiplicity
    energy    /= 2
    curvature /= 2
    surface   /= 3
    volume    /= 3
    return 2 * kappa * energy, surface, volume, curvature


if __name__ == "__main__":

    # unit sphere
    points, cells = meshzoo.icosa_sphere(8)
    tri = om.TriMesh(points, cells)
    om.write_mesh("test.stl", tri)

    m, s, v, c = calc_energy(tri, 1.0)
    print("Energy:", m)
    print("Surface: {} (4*pi={})".format(s, 4*np.pi))
    print("Volume: {} (4/3*pi={}".format(v, 4/3*np.pi))
    print("Curvature: {} (4*pi={})".format(c, 4*np.pi))

