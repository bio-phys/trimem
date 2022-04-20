#include <omp.h>
#include <random>
#include <chrono>
#include <stdexcept>

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"
#include "pybind11/iostream.h"
#include "pybind11/stl.h"

#include "defs.h"
#include "mesh.h"
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

PYBIND11_MODULE(core, m) {
    m.doc() = R"pbdoc(
        C++ library with python bindings for trimem.

        This module encapsulates the heavy lifting involved in the
        energy/gradient evaluations associated with trimem in a C++
        library offering bindings to be called from python.
    )pbdoc";

    // expose mesh
    expose_mesh(m);

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

    // (debug) energy stuff
    m.def(
        "gradient",
        &gradient,
        py::arg("mesh"),
        py::arg("estore"),
        py::arg("gradient"),
        py::arg("epsilon"),
        R"pbdoc(
        Finite difference gradient of energy.

        This is merely for testing/debugging. Use the gradient function
        exposed by the :class:`EnergyManager` class.

        Args:
            mesh (TriMesh): mesh instance.
            estore (EnergyManager): energy evaluation.
            gradient (numpy.ndarray): (N,3) array filled with the gradient.
            epsilon (float): finite difference perturbation magnitude.

        )pbdoc"
    );

    m.def(
        "area",
        &area,
        py::arg("mesh"),
        R"pbdoc(
        Compute surface area.

        Args:
            mesh (TriMesh): input mesh to be evaluated

        Returns:
            The value of the surface area of ``mesh``.
        )pbdoc"
   );

    m.def(
        "edges_length",
        &edges_length,
        py::arg("mesh"),
        R"pbdoc(
        Compute cumulative edge length.

        Args:
            mesh (TriMesh): input mesh to be evaluated

        Returns:
            The cumulated value of the length of all edges in ``mesh``.
        )pbdoc"
    );

    m.def(
        "avg_tri_props",
        &mean_tri_props,
        py::arg("mesh"),
        R"pbdoc(
        Average triangle area and edge length

        Args:
            mesh (TriMesh): mesh to process.

        Returns:
            A tuple (`a`, `l`) with `a` being the ``mesh``'s average face
            area and `l` being the mesh's average edge length. (Used for
            automatic detection of the characteristic lengths involved in
            the tether and repulsion penalties.)
        )pbdoc"
    );
}

}
