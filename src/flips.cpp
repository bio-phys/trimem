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

namespace trimem {

typedef std::chrono::high_resolution_clock myclock;
static std::mt19937 generator_(myclock::now().time_since_epoch().count());

int flip_serial(TriMesh& mesh, EnergyManager& estore, const real& flip_ratio)
{
    if (flip_ratio > 1.0)
        std::runtime_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // get initial vertex properties
    VertexProperties props = estore.properties(mesh);
    real             e0    = estore.energy(props);

    // acceptance probability distribution
    std::uniform_real_distribution<real> accept(0.0, 1.0);

    // proposal distribution
    std::uniform_int_distribution<int> propose(0, nedges-1);

    int acc = 0;
    for (int i=0; i<nflips; i++)
    {
        int  idx = propose(generator_);
        auto eh  = mesh.edge_handle(idx);
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

int flip_parallel_batches(TriMesh& mesh, EnergyManager& estore, const real& flip_ratio)
{
    if (flip_ratio > 1.0)
        throw std::range_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // get initial energy and associated vertex properties
    VertexProperties props = estore.properties(mesh);
    real             e0    = estore.energy(props);

    // set-up locks on edges
    std::vector<omp_lock_t> l_edges(nedges);
    for (auto& lock: l_edges)
        omp_init_lock(&lock);

    int acc  = 0;
#pragma omp parallel reduction(+:acc)
    {
        int ithread = omp_get_thread_num();
        int nthread = omp_get_num_threads();

        int itime   = myclock::now().time_since_epoch().count();
        std::mt19937 prng((ithread + 1) * itime);
        std::uniform_real_distribution<real> accept(0.0, 1.0);
        std::uniform_int_distribution<int> propose(0, nedges-1);
        int iflips = (int) std::ceil(nflips / nthread);

        for (int i=0; i<iflips; i++)
        {
#pragma omp barrier
            EdgeHandle eh(-1);

            {
                std::vector<OmpGuard> guards;
                int idx = propose(prng);
                eh = mesh.edge_handle(idx);
                guards = test_patch(mesh, idx, l_edges);
#pragma omp barrier
                if (guards.empty())
                {
                    continue;
                }
            }
            // here all locks will have been released

            if (not mesh.is_flip_ok(eh))
                continue;

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






/* BEGIN NSR VARIANT */


int flip_serial_nsr(TriMesh& mesh, EnergyManagerNSR& estore, const real& flip_ratio)
{
    if (flip_ratio > 1.0)
        std::runtime_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // get initial vertex properties
    VertexPropertiesNSR props = estore.properties(mesh);
    real             e0    = estore.energy(props);

    // acceptance probability distribution
    std::uniform_real_distribution<real> accept(0.0, 1.0);

    // proposal distribution
    std::uniform_int_distribution<int> propose(0, nedges-1);

    int acc = 0;
    for (int i=0; i<nflips; i++)
    {
        int  idx = propose(generator_);
        auto eh  = mesh.edge_handle(idx);
        if (mesh.is_flip_ok(eh) and !mesh.is_boundary(eh))
        {
            // remove old properties
            auto oprops = edge_vertex_properties_nsr(mesh, eh, *(estore.bonds)
                                                 );
            props -= oprops;

            // update with new properties
            mesh.flip(eh);
            auto nprops = edge_vertex_properties_nsr(mesh, eh, *(estore.bonds)
                                                 );
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

int flip_parallel_batches_nsr(TriMesh& mesh, EnergyManagerNSR& estore, const real& flip_ratio)
{
    if (flip_ratio > 1.0)
        throw std::range_error("flip_ratio must be <= 1.0");

    int nedges = mesh.n_edges();
    int nflips = (int) (nedges * flip_ratio);

    // get initial energy and associated vertex properties
    VertexPropertiesNSR props = estore.properties(mesh);
    real             e0    = estore.energy(props);

    // set-up locks on edges
    std::vector<omp_lock_t> l_edges(nedges);
    for (auto& lock: l_edges)
        omp_init_lock(&lock);

    int acc  = 0;
#pragma omp parallel reduction(+:acc)
    {
        int ithread = omp_get_thread_num();
        int nthread = omp_get_num_threads();

        int itime   = myclock::now().time_since_epoch().count();
        std::mt19937 prng((ithread + 1) * itime);
        std::uniform_real_distribution<real> accept(0.0, 1.0);
        std::uniform_int_distribution<int> propose(0, nedges-1);
        int iflips = (int) std::ceil(nflips / nthread);

        for (int i=0; i<iflips; i++)
        {
#pragma omp barrier
            EdgeHandle eh(-1);

            {
                std::vector<OmpGuard> guards;
                int idx = propose(prng);
                eh = mesh.edge_handle(idx);
                guards = test_patch(mesh, idx, l_edges);
#pragma omp barrier
                if (guards.empty())
                {
                    continue;
                }
            }
            // here all locks will have been released

            if (not mesh.is_flip_ok(eh))
                continue;

            // compute differential properties
            auto dprops = edge_vertex_properties_nsr(mesh, eh, *(estore.bonds)
                                                 );
            mesh.flip(eh);
            dprops -= edge_vertex_properties_nsr(mesh, eh, *(estore.bonds)
                                             );

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


/* END NSR VARIANT */

}
