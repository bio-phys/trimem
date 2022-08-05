/** \file mesh.h
 * \brief Mesh definition.
 *
 * This just defines the TriMesh as does openmesh-python. See mesh_py.cpp for
 * the exposure to python.
 */
#ifndef MESH_H
#define MESH_H

#include <OpenMesh/Core/IO/MeshIO.hh>
#include <OpenMesh/Core/Mesh/TriMesh_ArrayKernelT.hh>

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

}
#endif
