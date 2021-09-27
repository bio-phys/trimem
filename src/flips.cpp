/** \file flips.cpp
 * \brief Performing edge flips on openmesh.
 */
#include <algorithm>
#include <chrono>
#include <random>

#include "defs.h"

#include "flips.h"
#include "energy.h"
#include "mesh_properties.h"

#include <pybind11/stl.h>

namespace trimem {

typedef std::chrono::high_resolution_clock myclock;
static std::mt19937 generator_(myclock::now().time_since_epoch().count());

VertexProperties edge_vertex_properties(TriMesh& mesh,
                                       const EdgeHandle& eh,
                                       const BondPotential& bonds)
{
    VertexProperties props{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    for (int i=0; i<2; i++)
    {
        // vertex properties of the first face
        auto heh = mesh.halfedge_handle(eh, i);

        auto ve = mesh.to_vertex_handle(heh);
        props += vertex_properties(mesh, bonds, ve);

        auto next_heh = mesh.next_halfedge_handle(heh);
        ve = mesh.to_vertex_handle(next_heh);
        props += vertex_properties(mesh, bonds, ve);
    }

    return props;
}

int flip_serial(TriMesh& mesh, EnergyManager& estore, real& flip_ratio)
{
    if (flip_ratio > 1.0)
        std::runtime_error("flip_ratio must be <= 1.0");

    int nflips = (int) (mesh.n_edges() * flip_ratio);

    // generate random vector of edge indices
    std::vector<size_t> idx;
    idx.reserve(mesh.n_edges());
    for (size_t i=0; i<mesh.n_edges(); i++) idx.push_back(i);
    std::shuffle(idx.begin(), idx.end(), generator_);

    // get initial vertex properties
    real e0 = estore.energy();
    VertexProperties props = estore.properties;

    // acceptance probability distribution
    std::uniform_real_distribution<real> accept(0.0,1.0);

    int acc = 0;
    for (int i=0; i<nflips; i++)
    {
        auto eh = mesh.edge_handle(idx[i]);
        if (mesh.is_flip_ok(eh) and !mesh.is_boundary(eh))
        {
            // remove old properties
            auto oprops = edge_vertex_properties(mesh, eh, *(estore.bonds));
            props -= oprops;

            // update with new properties
            mesh.flip(eh);
            auto nprops = edge_vertex_properties(mesh, eh, *(estore.bonds));
            props += nprops;

            // evaluate energy
            real en = estore.energy(props);
            real de = en - e0;

            // evaluate acceptance probability
            real alpha = de < 0.0 ? 1.0 : std::exp(-de);
            real u     = accept(generator_);
            if (u <= alpha)
            {
                e0 = en;
                acc += 1;
            }
            else
            {
                mesh.flip(eh);
                props -= nprops;
                props += oprops;
            }
        }
    }

    return acc;
}


void expose_flips(py::module& m)
{
    m.def("flip", &flip_serial);
}

}
