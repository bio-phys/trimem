#include <omp.h>
#include <random>
#include <chrono>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/iostream.h"
#include "pybind11/stl.h"

#include "defs.h"
#include "mesh_util.h"
#include "energy.h"
#include "flips.h"
#include "mesh_repulsion.h"
#include "nlists/nlist.h"

namespace trimem {

real area(const TriMesh& mesh)
{
    real area = 0.0;

    #pragma omp parallel for reduction(+:area)
    for (size_t i=0; i<mesh.n_faces(); i++)
    {
        auto he = mesh.halfedge_handle(mesh.face_handle(i));
        area += face_area(mesh, he);
    }

    return area;
}

real edges_length(const TriMesh& mesh)
{
    real length = 0.0;

    #pragma omp parallel for reduction(+:length)
    for (size_t i=0; i<mesh.n_edges(); i++)
    {
        auto he = mesh.halfedge_handle(mesh.edge_handle(i),0);
        length += edge_length(mesh, he);
    }

    return length;
}

std::tuple<real, real> mean_tri_props(const TriMesh& mesh)
{
    return std::make_tuple<real, real>(area(mesh)/mesh.n_faces(),
                                       edges_length(mesh)/mesh.n_edges());
}

void gradient(TriMesh& mesh,
              EnergyManager& estore,
              py::array_t<real>& grad,
              real eps=1.0e-6)
{
    // unperturbed energy
    real e0 = estore.energy(mesh);

    auto r_grad = grad.mutable_unchecked<2>();
    for (int i=0; i<mesh.n_vertices(); i++)
    {
        Point& point = mesh.point(mesh.vertex_handle(i));
        for (int j=0; j<3; j++)
        {
            // do perturbation
            point[j] += eps;

            // evaluate differential energy
            real de = ( estore.energy(mesh) - e0 ) / eps;
            r_grad(i,j) = de;

            // undo perturbation
            point[j] -= eps;
        }
    }
}

PYBIND11_MODULE(_core, m) {
    m.doc() = "Trimem python bindings";

    // expose mesh properties
    expose_properties(m);

    // expose mesh utils
    expose_mesh_utils(m);

    // expose parameters
    expose_parameters(m);

    // expose energy
    expose_energy(m);

    // expose flips
    expose_flips(m);

    // expose neighbour lists
    expose_nlists(m);

    // energy stuff
    m.def("gradient", &gradient, "Finite difference gradient of energy");

    m.def("area", &area, "TriMesh area");
    m.def("edges_length", &edges_length, "TriMesh edge length");
    m.def("avg_tri_props", &mean_tri_props, "Avg. triangle area/edge length");
}

}
