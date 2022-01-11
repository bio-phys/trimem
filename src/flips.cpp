/** \file flips.cpp
 * \brief Performing edge flips on openmesh.
 */
#include <omp.h>
#include <algorithm>
#include <chrono>
#include <random>

#include "defs.h"

#include "flips.h"
#include "flip_utils.h"
#include "energy.h"
#include "mesh_properties.h"
#include "omp_guard.h"

#include <pybind11/stl.h>

namespace trimem {

typedef std::chrono::high_resolution_clock myclock;
static std::mt19937 generator_(myclock::now().time_since_epoch().count());

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
            auto oprops = edge_vertex_properties(mesh, eh, *(estore.bonds),
                                                 *(estore.repulse));
            props -= oprops;

            // update with new properties
            mesh.flip(eh);
            auto nprops = edge_vertex_properties(mesh, eh, *(estore.bonds),
                                                 *(estore.repulse));
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

int flip_parallel_batches(TriMesh& mesh, EnergyManager& estore, real& flip_ratio)
{
    if (flip_ratio > 1.0)
        throw std::range_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // get initial energy and associated vertex properties
    real e0 = estore.energy();
    VertexProperties props = estore.properties;

    // set-up lock on edges
    std::vector<OmpGuard> l_edges(nedges);

    int acc  = 0;
#pragma omp parallel reduction(+:acc)
    {
        int ithread = omp_get_thread_num();
        int nthread = omp_get_num_threads();

        int itime   = myclock::now().time_since_epoch().count();
        std::mt19937 prng((ithread + 1) * itime);
        std::uniform_real_distribution<real> accept(0.0,1.0);

        //  get this thread's edges
        int iedges = (int) std::ceil(nedges / nthread);
        int ilow   = ithread * iedges;
        int iupp   = (ithread + 1) * iedges;
        iupp   = iupp < nedges ? iupp : nedges;
        iedges = iupp > ilow ? iupp - ilow : 0;
        std::vector<int> edges;
        edges.reserve(iedges);
        for (int i=ilow; i<iupp; i++)
        {
            edges.push_back(i);
        }

        // shuffle locally
        std::shuffle(edges.begin(), edges.end(), prng);

        // result vector
        int iflips = (int) std::ceil(nflips / nthread);

        for (int i=0; i<iflips; i++)
        {
#pragma omp barrier
            bool locked = false;
            EdgeHandle eh(-1);
            if (i<iedges)
            {
                int idx = edges[i];
                eh = mesh.edge_handle(idx);
                locked = test_guards(mesh, idx, l_edges);
            }
#pragma omp barrier
            if (locked)
            {
                auto patch = flip_patch(mesh, eh);
                for (const int& i: patch) l_edges[i].release();
            }
            else
            {
                continue;
            }

            // compute differential properties
            auto dprops = edge_vertex_properties(mesh, eh, *(estore.bonds),
                                                 *(estore.repulse));
            mesh.flip(eh);
            dprops -= edge_vertex_properties(mesh, eh, *(estore.bonds),
                                             *(estore.repulse));

            real u = accept(prng);

            // evaluate energy
#pragma omp critical
            {
                props -= dprops;
                real en  = estore.energy(props);
                real de = en - e0;

                // evaluate acceptance probability
                real alpha = de < 0.0 ? 1.0 : std::exp(-de);
                if (u <= alpha)
                {
                    e0 = en;
                    acc += 1;
                }
                else
                {
                    mesh.flip(eh);
                    props += dprops;
                }
            }
        }
    } // parallel

    return acc;
}

void expose_flips(py::module& m)
{
    m.def("flip", &flip_serial);
    m.def("pflip", &flip_parallel_batches);
}

}
