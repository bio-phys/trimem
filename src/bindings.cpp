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

namespace trimem {

real area(TriMesh& mesh)
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

void gradient(TriMesh& mesh,
              EnergyManager& estore,
              py::array_t<real>& grad,
              real eps=1.0e-6)
{
    // unperturbed energy
    real e0 = estore.energy();

    auto r_grad = grad.mutable_unchecked<2>();
    for (int i=0; i<mesh.n_vertices(); i++)
    {
        Point& point = mesh.point(mesh.vertex_handle(i));
        for (int j=0; j<3; j++)
        {
            // do perturbation
            point[j] += eps;

            // evaluate differential energy
            real de = ( estore.energy() - e0 ) / eps;
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

    // energy stuff
    m.def("gradient", &gradient, "Finite difference gradient of energy");

    m.def("area", &area, "TriMesh area");
}

}
