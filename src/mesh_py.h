/** \file mesh_py.h
 * \brief Python utilities for TriMesh.
 *
 * Minimal subset of the the python exposure of openmesh-python.
 */
#ifndef MESH_PY_H
#define MESH_PY_H

#include "mesh.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

namespace py = pybind11;

namespace trimem {

// construct mesh from points and cells
// this is the add_vertices and add_face functions from openmesh
TriMesh
from_points_cells(py::array_t<typename TriMesh::Point::value_type> points,
                  py::array_t<int>                                  cells);

// get faces from mesh (memory maintenance goes to python)
py::array_t<int> fv_indices(TriMesh& mesh);



// get reference to mesh-points (memory remains with the mesh)
py::array_t<typename TriMesh::Point::value_type> points(TriMesh& mesh);
}
#endif
