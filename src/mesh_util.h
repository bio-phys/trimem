/** \file mesh_util.h
 * \brief Geometric properties and gradients on a OpenMesh::TriMesh.
 */
#ifndef MESH_UTIL_H
#define MESH_UTIL_H

#include "defs.h"

#include "OpenMesh/Core/Geometry/VectorT.hh"

namespace trimem {

inline TriMesh::Normal edge_vector(const TriMesh& mesh,
                                   const HalfedgeHandle& he)
{
    TriMesh::Normal edge;
    mesh.calc_edge_vector(he, edge);
    return edge;
}

inline TriMesh::Normal face_normal(const TriMesh& mesh,
                                   const HalfedgeHandle& he)
{
    TriMesh::Normal normal;
    mesh.calc_sector_normal(he, normal);
    return normal/OpenMesh::norm(normal);
}

inline real edge_length(const TriMesh& mesh, const HalfedgeHandle& he)
{
    return OpenMesh::norm(edge_vector(mesh, he));
}

inline std::vector<Point> edge_length_grad(const TriMesh& mesh,
                                           const HalfedgeHandle& he)
{
    auto edge = edge_vector(mesh, he);
    edge /= OpenMesh::norm(edge);

    std::vector<Point> gradient(2, edge);
    gradient[0] *= -1;

    return gradient;
}

inline real face_area(const TriMesh& mesh, const HalfedgeHandle& he)
{
    TriMesh::Normal normal;
    mesh.calc_sector_normal(he, normal);
    return OpenMesh::norm(normal)/2;
}

inline std::vector<Point> face_area_grad(const TriMesh& mesh,
                                        const HalfedgeHandle& he)
{
    auto normal = face_normal(mesh, he);
    normal /= OpenMesh::norm(normal);

    auto e0 = edge_vector(mesh, he);
    auto e1 = edge_vector(mesh, mesh.next_halfedge_handle(he));
    auto e2 = edge_vector(mesh, mesh.prev_halfedge_handle(he));

    std::vector<Point> gradient;
    gradient.reserve(3);

    gradient.push_back( 0.5 * OpenMesh::cross(normal, e1) );
    gradient.push_back( 0.5 * OpenMesh::cross(normal, e2) );
    gradient.push_back( 0.5 * OpenMesh::cross(normal, e0) );

    return gradient;
}

inline real face_volume(const TriMesh& mesh, const HalfedgeHandle& he)
{
    auto p0 = mesh.point(mesh.from_vertex_handle(he));
    auto p1 = mesh.point(mesh.to_vertex_handle(he));
    auto p2 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(he)));

    return OpenMesh::cross(p1,p2).dot(p0) / 6;
}

inline std::vector<Point> face_volume_grad(const TriMesh& mesh,
                                           const HalfedgeHandle& he)
{
    auto p0 = mesh.point(mesh.from_vertex_handle(he));
    auto p1 = mesh.point(mesh.to_vertex_handle(he));
    auto p2 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(he)));

    std::vector<Point> gradient;
    gradient.reserve(3);

    gradient.push_back( OpenMesh::cross(p1,p2) / 6 );
    gradient.push_back( OpenMesh::cross(p2,p0) / 6 );
    gradient.push_back( OpenMesh::cross(p0,p1) / 6 );

    return gradient;
}

inline real dihedral_angle(const TriMesh& mesh, const HalfedgeHandle& he)
{
    return mesh.calc_dihedral_angle(he);
}

inline std::vector<Point> dihedral_angle_grad(const TriMesh& mesh,
                                              const HalfedgeHandle& he)
{
    auto n0 = face_normal(mesh, he);
    auto n1 = face_normal(mesh, mesh.opposite_halfedge_handle(he));
    real l  = edge_length(mesh, he);

    real a01 = mesh.calc_sector_angle(mesh.prev_halfedge_handle(he));
    real a03 = mesh.calc_sector_angle(he);
    real a02 = mesh.calc_sector_angle(mesh.opposite_halfedge_handle(he));
    real a04 = mesh.calc_sector_angle(mesh.prev_halfedge_handle(
                                        mesh.opposite_halfedge_handle(he)));

    std::vector<Point> gradient;
    gradient.reserve(4);

    gradient.push_back(  (1./std::tan(a03) * n0 + 1./std::tan(a04) * n1 ) / l );
    gradient.push_back(  (1./std::tan(a01) * n0 + 1./std::tan(a02) * n1 ) / l );
    gradient.push_back( -(1./std::tan(a01) + 1./std::tan(a03)) * n0 / l );
    gradient.push_back( -(1./std::tan(a02) + 1./std::tan(a04)) * n1 / l );

    return gradient;
}

}
#endif
