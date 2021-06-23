/** \file mesh_math.h
 * \brief Geomoetric utility function on edges.
 */
#ifndef MESH_MATH_H
#define MESH_MATH_H

#include <ctgmath>

#include "MeshTypes.hh"
#include "OpenMesh/Core/Geometry/VectorT.hh"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

typedef double real;

namespace trimem {

typedef OpenMesh::HalfedgeHandle HalfedgeHandle;

//! Gradient
typedef TriMesh::Point Vector;

TriMesh::Normal edge_vector(TriMesh& mesh, HalfedgeHandle he)
{
    TriMesh::Normal edge;
    mesh.calc_edge_vector(he, edge);
    return edge;
}

TriMesh::Normal face_normal(TriMesh& mesh, HalfedgeHandle he)
{
    TriMesh::Normal normal;
    mesh.calc_sector_normal(he, normal);
    return normal/OpenMesh::norm(normal);
}

real edge_length(TriMesh& mesh, HalfedgeHandle& he)
{
    return OpenMesh::norm(edge_vector(mesh, he));
}

std::vector<Vector> edge_length_grad(TriMesh& mesh, HalfedgeHandle& he)
{
    std::vector<Vector> gradient;
    gradient.reserve(2);

    auto edge = edge_vector(mesh, he);
    edge /= OpenMesh::norm(edge);

    gradient.push_back( -edge );
    gradient.push_back(  edge );

    return gradient;
}

real face_area(TriMesh& mesh, HalfedgeHandle he)
{
    TriMesh::Normal normal;
    mesh.calc_sector_normal(he, normal);
    return OpenMesh::norm(normal)/2;
}

std::vector<Vector> face_area_grad(TriMesh& mesh, HalfedgeHandle he)
{
    std::vector<Vector> gradient;
    gradient.reserve(3);

    auto normal = face_normal(mesh, he);

    auto e0 = edge_vector(mesh, he);
    auto e1 = edge_vector(mesh, mesh.next_halfedge_handle(he));
    auto e2 = edge_vector(mesh, mesh.prev_halfedge_handle(he));

    gradient.push_back( 0.5 * OpenMesh::cross(normal, e1) );
    gradient.push_back( 0.5 * OpenMesh::cross(normal, e2) );
    gradient.push_back( 0.5 * OpenMesh::cross(normal, e0) );

    return gradient;
}

real face_volume(TriMesh& mesh, HalfedgeHandle he)
{
    auto p0 = mesh.point(mesh.from_vertex_handle(he));
    auto p1 = mesh.point(mesh.to_vertex_handle(he));
    auto p2 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(he)));

    return OpenMesh::cross(p1,p2).dot(p0) / 6;
}

std::vector<Vector> face_volume_grad(TriMesh& mesh, HalfedgeHandle he)
{
    std::vector<Vector> gradient;
    gradient.reserve(3);

    auto p0 = mesh.point(mesh.from_vertex_handle(he));
    auto p1 = mesh.point(mesh.to_vertex_handle(he));
    auto p2 = mesh.point(mesh.to_vertex_handle(mesh.next_halfedge_handle(he)));

    gradient.push_back( OpenMesh::cross(p1,p2) / 6 );
    gradient.push_back( OpenMesh::cross(p2,p0) / 6 );
    gradient.push_back( OpenMesh::cross(p0,p1) / 6 );

    return gradient;
}

real dihedral_angle(TriMesh& mesh, HalfedgeHandle& he)
{
    return mesh.calc_dihedral_angle(he);
}

std::vector<Vector> dihedral_angle_grad(TriMesh& mesh, HalfedgeHandle& he)
{
    std::vector<Vector> gradient;
    gradient.reserve(4);

    auto n0 = face_normal(mesh, he);
    auto n1 = face_normal(mesh, mesh.opposite_halfedge_handle(he));
    real l  = edge_length(mesh, he);

    real a01 = mesh.calc_sector_angle(mesh.prev_halfedge_handle(he));
    real a03 = mesh.calc_sector_angle(he);
    real a02 = mesh.calc_sector_angle(mesh.opposite_halfedge_handle(he));
    real a04 = mesh.calc_sector_angle(mesh.prev_halfedge_handle(
                                        mesh.opposite_halfedge_handle(he)));

    gradient.push_back(  (1./std::tan(a03) * n0 + 1./std::tan(a04) * n1 ) / l );
    gradient.push_back(  (1./std::tan(a01) * n0 + 1./std::tan(a02) * n1 ) / l );
    gradient.push_back( -(1./std::tan(a01) + 1./std::tan(a03)) * n0 / l );
    gradient.push_back( -(1./std::tan(a02) + 1./std::tan(a04)) * n1 / l );

    return gradient;
}

// vector of some Row-type to numpy
template<class Row>
py::array_t<typename Row::value_type> tonumpy(Row& _vec, size_t _n = 1) {
	typedef typename Row::value_type dtype;
	std::vector<size_t> shape;
	std::vector<size_t> strides;
	if (_n == 1) {
		shape = {_vec.size()};
		strides = {sizeof(dtype)};
	}
	else {
		shape = {_n, _vec.size()};
		strides = {_vec.size() * sizeof(dtype), sizeof(dtype)};
	}
	return py::array_t<dtype>(shape, strides, _vec.data());
}

void expose_properties(py::module& m)
{
    m.def("edge_length", &edge_length);
    m.def("edge_length_grad", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = edge_length_grad(mesh, he);
            return tonumpy(res[0], res.size());});

    m.def("face_area", &face_area);
    m.def("face_area_grad", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_area_grad(mesh, he);
            return tonumpy(res[0], res.size());});

    m.def("face_volume", &face_volume);
    m.def("face_volume_grad", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_volume_grad(mesh, he);
            return tonumpy(res[0], res.size());});

    m.def("dihedral_angle", &dihedral_angle);
    m.def("dihedral_angle_grad", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = dihedral_angle_grad(mesh, he);
            return tonumpy(res[0], res.size());});
}

}
#endif
