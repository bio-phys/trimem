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

//! a row-vector of the derivation of a scalar wrt components of another vector
//! since the 'other' vector here is always a point coordinate, the gradient
//! can be of type 'TriMesh::Point', i.e., it has lenght 3.
typedef TriMesh::Point Vector;

//! a MxN jacobian matrix of the gradients of an M-vector quantity wrt an
//! N-vector (here 3-vector)
typedef std::vector<Vector> Matrix;

// Matrix vector product
Vector dot(Matrix& A, Vector& x)
{
    Vector y;
    for (size_t i=0; i<A.size(); i++)
    {
        y[i] = A[i].dot(x);
    }

    return y;
}

real edge_length(TriMesh& mesh, HalfedgeHandle& he)
{
    return mesh.calc_edge_length(he);
}

std::vector<Vector> edge_length_grad(TriMesh& mesh, HalfedgeHandle& he)
{
    std::vector<Vector> gradient;
    gradient.reserve(2);

    real length = edge_length(mesh, he);
    auto to     = mesh.to_vertex_handle(he);
    auto from   = mesh.from_vertex_handle(he);
    TriMesh::Point diff = ( mesh.point(to) - mesh.point(from) );

    diff /= length;
    gradient.push_back( -diff );
    gradient.push_back( diff );

    return gradient;
}

TriMesh::Normal face_centroid(TriMesh& mesh, HalfedgeHandle& he)
{
    TriMesh::Point center = {0, 0, 0};
    center += mesh.point( mesh.from_vertex_handle(he) );
    center += mesh.point( mesh.to_vertex_handle(he) );
    center += mesh.point( mesh.to_vertex_handle(mesh.next_halfedge_handle(he)) );
    center /= 3;
    return center;
}

std::vector<Matrix> face_centroid_grad(TriMesh& mesh, HalfedgeHandle he)
{
    std::vector<Matrix> jacs;
    jacs.reserve(3);

    Matrix m = { {1./3, 0., 0.}, {0., 1./3, 0.}, {0., 0., 1./3} };

    jacs.push_back(m);
    jacs.push_back(m);
    jacs.push_back(m);

    return jacs;
}

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
    return normal;
}

std::vector<Matrix> face_normal_grad(TriMesh& mesh, HalfedgeHandle he)
{
    std::vector<Matrix> jacs;
    jacs.reserve(3);

    auto x0 = mesh.point( mesh.from_vertex_handle(he) );
    auto x1 = mesh.point( mesh.to_vertex_handle(he));
    auto x2 = mesh.point( mesh.to_vertex_handle(mesh.next_halfedge_handle(he)) );

    // dn/dx0
    auto d21 = x2-x1;
    jacs.push_back({ {       0,  -d21[2],  d21[1] },
                     {  d21[2],       0,  -d21[0] },
                     { -d21[1],   d21[0],       0  } });

    // dn/dx1
    auto d10 = x0-x1;
    jacs.push_back({ {              0,   d21[2]-d10[2],  d10[1]-d21[1] },
                     { -d21[2]+d10[2],               0,  d21[0]-d10[0] },
                     { -d10[1]+d21[1],  -d21[0]+d10[0],              0 } });

    // dn/dx2
    jacs.push_back({ {       0,  d10[2], -d10[1] },
                     { -d10[2],       0,  d10[0] },
                     {  d10[1], -d10[0],       0 } });

    return jacs;

}

real face_area(TriMesh& mesh, HalfedgeHandle he)
{
    auto normal = face_normal(mesh, he);
    return OpenMesh::norm(normal)/2;
}

std::vector<Vector> face_area_grad(TriMesh& mesh, HalfedgeHandle he)
{
    std::vector<Vector> gradient;
    gradient.reserve(3);

    auto normal = face_normal(mesh, he);
    real nn     = OpenMesh::norm(normal)*2.0;

    // (unnormalized) face normal gradients
    auto dn = face_normal_grad(mesh, he);

    // assemble gradient wrt to the 3 face vertices
    Vector one(1.0);
    //x0, x1 and x2
    gradient.push_back( -dot(dn[0], normal) / nn);
    gradient.push_back( -dot(dn[1], normal) / nn);
    gradient.push_back( -dot(dn[2], normal) / nn);

    return gradient;
}

std::vector<Vector> face_area_grad_alt(TriMesh& mesh, HalfedgeHandle he)
{
    std::vector<Vector> gradient;
    gradient.reserve(3);

    auto normal = face_normal(mesh, he);
    normal /= OpenMesh::norm(normal);

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

std::vector<Vector> dihedral_angle_grad_alt(TriMesh& mesh, HalfedgeHandle& he)
{
    std::vector<Vector> gradient;
    gradient.reserve(4);

    auto n0 = face_normal(mesh, he);
    auto n1 = face_normal(mesh, mesh.opposite_halfedge_handle(he));
    auto ve = edge_vector(mesh, he);
    real l  = OpenMesh::norm(ve);

    n0 = n0/OpenMesh::norm(n0);
    n1 = n1/OpenMesh::norm(n1);

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

std::vector<Vector> dihedral_angle_grad(TriMesh& mesh, HalfedgeHandle& he)
{
    std::vector<Vector> gradient;
    gradient.reserve(4);

    auto n0 = face_normal(mesh, he);
    auto n1 = face_normal(mesh, mesh.opposite_halfedge_handle(he));
    auto ve = edge_vector(mesh, he);

    auto nn0 = OpenMesh::norm(n0);
    auto nn1 = OpenMesh::norm(n1);

    auto denom = nn0*nn1;
    if (denom == TriMesh::Scalar(0))
    {
        gradient.push_back(Vector(0));
        gradient.push_back(Vector(0));
        gradient.push_back(Vector(0));
        gradient.push_back(Vector(0));
        return gradient;
    }

    auto nom = n0.dot(n1);

    // da_cos
    int sgn = cross(n0,n1).dot(ve) >=0 ? 1 : -1;
    real da_cos = nom/denom;

    // (unnormalized) face normal derivatives
    auto dn0 = face_normal_grad(mesh, he);
    auto dn1 = face_normal_grad(mesh, mesh.opposite_halfedge_handle(he));

    // face area derivative (factor 2 missing here. Account for it later)
    auto da0 = face_area_grad(mesh, he);
    auto da1 = face_area_grad(mesh, mesh.opposite_halfedge_handle(he));

    // assemble gradients wrt the 4 vertices associated to the edge
    real fac = 1. / ( std::sqrt(1. - da_cos * da_cos) * denom * denom ) * sgn;
    // dx0
    gradient.push_back(
        ( ( dot(dn0[0], n1) + dot(dn1[1], n0) ) * denom +
          ( da0[0] * nn1 + da1[1] * nn0 ) * nom * 2
        ) * fac );

    // dx1
    gradient.push_back(
        ( ( dot(dn0[1], n1) + dot(dn1[0], n0) ) * denom +
          ( da0[1] * nn1 + da1[0] * nn0 ) * nom * 2
        ) * fac );

    // dx2
    gradient.push_back(
        ( dot(dn0[2], n1) * denom +
          da0[2] * nn1 * nom * 2
        ) * fac );

    // dx3
    gradient.push_back(
        ( dot(dn1[2], n0)* denom +
          da1[2] * nn0 * nom * 2
        ) * fac );

    return gradient;
}

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

    m.def("face_centroid", [](TriMesh& mesh, HalfedgeHandle& he){
            auto res = face_centroid(mesh, he);
            return tonumpy(res);});
    m.def("face_centroid_grad", [](TriMesh& mesh, HalfedgeHandle& he){
            auto res = face_centroid_grad(mesh, he);
            std::vector<py::array_t<typename Vector::value_type>> grad;
            for (size_t i=0; i<res.size(); i++)
                grad.push_back(tonumpy(res[i][0],res[i].size()));
            return grad;});

    m.def("face_normal", [](TriMesh& mesh, HalfedgeHandle& he){
            auto res = face_normal(mesh, he);
            return tonumpy(res);});
    m.def("face_normal_grad", [](TriMesh& mesh, HalfedgeHandle& he){
            auto res = face_normal_grad(mesh, he);
            std::vector<py::array_t<typename Vector::value_type>> grad;
            for (size_t i=0; i<res.size(); i++)
                grad.push_back(tonumpy(res[i][0],res[i].size()));
            return grad;});

    m.def("face_area", &face_area);
    m.def("face_area_grad", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_area_grad(mesh, he);
            return tonumpy(res[0], res.size());});
    m.def("face_area_grad_alt", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_area_grad_alt(mesh, he);
            return tonumpy(res[0], res.size());});

    m.def("face_volume", &face_volume);
    m.def("face_volume_grad", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_volume_grad(mesh, he);
            return tonumpy(res[0], res.size());});

    m.def("dihedral_angle", &dihedral_angle);
    m.def("dihedral_angle_grad", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = dihedral_angle_grad(mesh, he);
            return tonumpy(res[0], res.size());});
    m.def("dihedral_angle_grad_alt", [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = dihedral_angle_grad_alt(mesh, he);
            return tonumpy(res[0], res.size());});
}

}
#endif
