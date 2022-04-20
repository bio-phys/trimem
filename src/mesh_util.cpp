/** \file mesh_util.cpp
 * \brief Geometric properties and gradients on a OpenMesh::TriMesh.
 */
#include "mesh_util.h"

#include "numpy_util.h"

namespace trimem {

void expose_mesh_utils(py::module& m)
{
    m.def(
        "edge_length",
        &edge_length,
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate edge length.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The lenght of the edge referred to by ``halfedge_handle``.
        )pbdoc"
    );

    m.def(
        "edge_length_grad",
        [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = edge_length_grad(mesh, he);
            return tonumpy(res[0], res.size());
        },
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate gradient of edge length.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The gradient of the edge length referred to by ``halfedge_handle``
            wrt.\ the coordinates of the vertices that are connected to
            ``halfedge_handle``.
        )pbdoc"
    );

    m.def(
        "face_area",
        &face_area,
        R"pbdoc(
        Evaluate face area.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The area of the face associated to ``halfedge_handle``.
        )pbdoc"
    );

    m.def(
        "face_area_grad",
        [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_area_grad(mesh, he);
            return tonumpy(res[0], res.size());
        },
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate gradient of face area.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The gradient of the area of the face that is associated to
            ``halfedge_handle`` wrt. the coordinates of the vertices of that
            face.
        )pbdoc"
    );

    m.def(
        "face_volume",
        &face_volume,
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate face volume.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The volume of the tetrahedron given by the face associated to
            ``halfedge_handle`` and the origin.
        )pbdoc"
    );

    m.def(
        "face_volume_grad",
        [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = face_volume_grad(mesh, he);
            return tonumpy(res[0], res.size());
        },
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate gradient of face volume.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The gradient of the volume of the tetrahedron given by the face
            associated to ``halfedge_handle`` and the origin wrt to the
            coordinates of the face's vertices.
        )pbdoc"
    );

    m.def(
        "dihedral_angle",
        &dihedral_angle,
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate dihedral angle.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The dihedral angle between the face associated to
            ``halfedge_handle`` and the face associated to the
            opposite halfedge_handle.
        )pbdoc"
    );

    m.def(
        "dihedral_angle_grad",
        [](TriMesh& mesh, HalfedgeHandle& he) {
            auto res = dihedral_angle_grad(mesh, he);
            return tonumpy(res[0], res.size());
        },
        py::arg("mesh"),
        py::arg("halfedge_handle"),
        R"pbdoc(
        Evaluate gradient of dihedral angle.

        Args:
            mesh (TriMesh): mesh containing ``halfedge_handle``.
            halfedge_handle (HalfedgeHandle): handle to halfedge of interest.

        Returns:
            The gradient of the dihedral angle between the face associated to
            ``halfedge_handle`` and the face associated to the opposite
            halfedge_handle wrt to the coordinates of both faces.
        )pbdoc"
    );
}
}
