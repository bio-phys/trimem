/** \file mesh.h
 * \brief Mesh definition.
 *
 * This just defines the TriMesh as does openmesh-python but with only
 * a minimal exposure to python. If mesh manipulation is sought after in
 * python openmesh-python can be used explicitly.
 */
#ifndef MESH_H
#define MESH_H

#define OM_STATIC_BUILD

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace trimem {

struct MeshTraits : public OpenMesh::DefaultTraits {
	/** Use double precision points */
	typedef OpenMesh::Vec3d Point;

	/** Use double precision normals */
	typedef OpenMesh::Vec3d Normal;

	/** Use RGBA colors */
	typedef OpenMesh::Vec4f Color;

	/** Use double precision texcoords */
	typedef double TexCoord1D;
	typedef OpenMesh::Vec2d TexCoord2D;
	typedef OpenMesh::Vec3d TexCoord3D;
};

typedef OpenMesh::TriMesh_ArrayKernelT<MeshTraits> TriMesh;

// read mesh from file
TriMesh read_mesh(const std::string fname);

// minimal exposure
void expose_mesh(py::module& m);
}
#endif
