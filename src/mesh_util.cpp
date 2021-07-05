/** \file mesh_util.cpp
 * \brief Geometric properties and gradients on a OpenMesh::TriMesh.
 */
#include "mesh_util.h"

#include "numpy_util.h"

namespace trimem {

void expose_mesh_utils(py::module& m)
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
