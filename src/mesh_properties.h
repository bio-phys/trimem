/** \file mesh_properties.h
 * \brief Geometric properties and gradients on a OpenMesh::TriMesh.
 */
#ifndef MESH_PROPERTIES_H
#define MESH_PROPERTIES_H

#include "MeshTypes.hh"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

typedef double real;

namespace trimem {

typedef OpenMesh::HalfedgeHandle HalfedgeHandle;
typedef TriMesh::Point Gradient;

TriMesh::Normal edge_vector(TriMesh& mesh, HalfedgeHandle he);

TriMesh::Normal face_normal(TriMesh& mesh, HalfedgeHandle he);

real edge_length(TriMesh& mesh, HalfedgeHandle& he);

std::vector<Gradient> edge_length_grad(TriMesh& mesh, HalfedgeHandle& he);

real face_area(TriMesh& mesh, HalfedgeHandle he);

std::vector<Gradient> face_area_grad(TriMesh& mesh, HalfedgeHandle he);

real face_volume(TriMesh& mesh, HalfedgeHandle he);

std::vector<Gradient> face_volume_grad(TriMesh& mesh, HalfedgeHandle he);

real dihedral_angle(TriMesh& mesh, HalfedgeHandle& he);

std::vector<Gradient> dihedral_angle_grad(TriMesh& mesh, HalfedgeHandle& he);

// vector of some Gradients to numpy
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

void expose_properties(py::module& m);

}
#endif
