/** \file mesh_math.h
 * \brief Geomoetric utility function on edges.
 */
#ifndef MESH_MATH_H
#define MESH_MATH_H

#include "MeshTypes.hh"

typedef double real;

namespace trimem {

real edge_length(TriMesh& mesh, TriMesh::HalfedgeHandle& he)
{
    return mesh.calc_edge_length(he);
}

real dihedral_angle(TriMesh& mesh, TriMesh::HalfedgeHandle& he)
{
    return mesh.calc_dihedral_angle(he);
}

TriMesh::Normal face_centroid(TriMesh& mesh, TriMesh::HalfedgeHandle& he)
{
    TriMesh::Point center = {0, 0, 0};
    center += mesh.point(mesh.from_vertex_handle(he));
    center += mesh.point(mesh.to_vertex_handle(he));
    center += mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(he)));
    center /= 3;
    return center;
}

TriMesh::Normal face_normal(TriMesh& mesh, TriMesh::HalfedgeHandle& he)
{
    TriMesh::Normal normal;
    mesh.calc_sector_normal(he, normal);
    return normal;
}

}
#endif
