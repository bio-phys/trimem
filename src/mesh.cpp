/** \file mesh.cpp
 */
#include "mesh.h"

#include "OpenMesh/Core/Mesh/Handles.hh"

#include "pybind11/numpy.h"

namespace trimem {

TriMesh read_mesh(const std::string fname)
{
    TriMesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, fname))
    {
        std::cerr << "read error on file " << fname << "\n";
        exit(1);
    }
    return mesh;
}

// construct mesh from points and cells
// this is the add_vertices and add_face functions from openmesh
TriMesh
from_points_cells(py::array_t<typename TriMesh::Point::value_type> points,
                  py::array_t<int>                                  cells)
{

    TriMesh mesh;

    // --------- add vertices
    // return if _points is empty
    if (points.size() == 0) {
      return mesh;
    }

    // points is not empty, throw if points has wrong shape
    if (points.ndim() != 2 || points.shape(1) != 3)
    {
        throw std::runtime_error("Array 'points' must have shape (n,3)");
    }

    auto proxy_p = points.unchecked<2>();

    for (ssize_t i = 0; i < proxy_p.shape(0); ++i)
    {
        mesh.add_vertex({proxy_p(i, 0), proxy_p(i, 1), proxy_p(i, 2)});
    }

    // --------- cells
    // return if two few points or empty cells
    if (mesh.n_vertices() < 3 || cells.size() == 0)
    {
      return mesh;
    }

    // cells is not empty, throw if faces has wrong shape
    if (cells.ndim() != 2 || cells.shape(1) != 3)
    {
        throw std::runtime_error("Array 'cells' must have shape (n,3)");
    }

    auto proxy_c = cells.unchecked<2>();

    for (ssize_t i = 0; i < proxy_c.shape(0); ++i)
    {
        std::vector<OpenMesh::VertexHandle> vhandles;
        for (ssize_t j = 0; j < cells.shape(1); ++j)
        {
            if (cells.at(i, j) >= 0 && cells.at(i, j) < mesh.n_vertices())
            {
                vhandles.push_back(OpenMesh::VertexHandle(proxy_c(i, j)));
            }
        }
        if (vhandles.size() >= 3)
        {
            mesh.add_face(vhandles);
        }
    }

    return mesh;
}

// get faces from mesh (memory maintenance goes to python)
py::array_t<int> fv_indices(TriMesh& mesh)
{
	if (mesh.n_faces() == 0)
  {
		return py::array_t<int>();
	}

  int len      = mesh.n_faces();
  auto indices = py::array_t<int>({len,3});
  auto proxy   = indices.mutable_unchecked<2>();

	for (int i=0; i<mesh.n_faces(); i++)
  {
      auto fh = mesh.face_handle(i);
      auto fv_it = mesh.fv_iter(fh);
      proxy(fh.idx(),0) = fv_it->idx(); ++fv_it;
      proxy(fh.idx(),1) = fv_it->idx(); ++fv_it;
      proxy(fh.idx(),2) = fv_it->idx();
	}
	return indices;
}

// get reference to mesh-points (memory remains with the mesh)
py::array_t<typename TriMesh::Point::value_type> points(TriMesh& mesh)
{
    typedef typename TriMesh::Point::value_type dtype;

    auto& point = mesh.point(OpenMesh::VertexHandle(0));

		std::vector<size_t> shape = {mesh.n_vertices(), point.size()};
		std::vector<size_t> strides = {point.size() * sizeof(dtype), sizeof(dtype)};

	  return py::array_t<dtype>(shape, strides, point.data(), py::cast(mesh));
}

void expose_mesh(py::module& m){

    // not really necessary but used in testing
    py::class_<OpenMesh::HalfedgeHandle>(m, "HalfedgeHandle")
        .def(py::init());

    py::class_<TriMesh>(m, "TriMesh")
        .def(py::init())
		    .def(py::init([](
            py::array_t<typename TriMesh::Point::value_type> points,
            py::array_t<int>                                  faces
            )
            {
                TriMesh mesh;
                return from_points_cells(points, faces);
			      }
        ), py::arg("points"), py::arg("faces"))
        .def("fv_indices", &fv_indices)
        .def("points", &points)
        .def("n_vertices", [](TriMesh& mesh) {return mesh.n_vertices();})
        .def("n_edges", [](TriMesh& mesh) {return mesh.n_edges();})
        .def("halfedge_handle", [](TriMesh& mesh, int i)
            {
                return mesh.halfedge_handle(i);
            }
        );

    m.def("read_mesh", &read_mesh);
}
}
