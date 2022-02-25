/** \file defs.h
 * \brief Common type definitions.
 */
#ifndef DEFS_H
#define DEFS_H

#include "mesh.h"

#include "OpenMesh/Core/Mesh/Handles.hh"

namespace trimem {

typedef TriMesh::Point Point;
typedef TriMesh::Point::value_type real;

typedef OpenMesh::HalfedgeHandle HalfedgeHandle;
typedef OpenMesh::VertexHandle   VertexHandle;
typedef OpenMesh::EdgeHandle     EdgeHandle;

}
#endif
