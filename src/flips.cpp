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
    VertexProperties props = estore.properties(mesh);
    real             e0    = estore.energy(props);

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
            EdgeHandle eh(-1);

            {
                std::vector<OmpGuard> guards;
                if (i<iedges)
                {
                    int idx = edges[i];
                    eh = mesh.edge_handle(idx);
                    guards = test_patch(mesh, idx, l_edges);
                }
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

void expose_flips(py::module& m)
{
    m.def(
        "flip",
        &flip_serial,
        py::arg("mesh"),
        py::arg("estore"),
        py::arg("flip_ratio"),
        R"pbdoc(
        Serial flip sweep.

        Performs a sweep over a fraction ``flip_ratio`` of edges in ``mesh``
        trying to flip each edge sequentially and evaluating the energies
        associated to the flip against the Metropolis criterion.

        Args:
            mesh (TriMesh): input mesh to be used
            estore (EnergyManager): instance of :class:`EnergyManager` used in
                combination with the ``mesh`` to evaluate energy differences
                necessary for flip acceptance/rejection.
            flip_ratio (float): ratio of edges to test (must be in [0,1]).
        )pbdoc"
    );

    m.def(
        "pflip",
        &flip_parallel_batches,
        py::arg("mesh"),
        py::arg("estore"),
        py::arg("flip_ratio"),
        R"pbdoc(
        Batch parallel flip sweep.

        Performs a sweep over a fraction ``flip_ratio`` of edges in ``mesh``
        in a batch parallel fashion albeit maintaining chain ergodicity.
        To this end, a batch of edges is selected at random by each thread
        from a thread-local pool of edges. If an edge is free to be flipped
        independently, i.e., no overlap of its patch with the patch of other
        edges (realized by a locking mechanism), it is flipped and its
        differential contribution to the Hamiltonian is evaluated in parallel
        for the whole batch. Ergodicity is maintained by subsequently evaluating
        the Metropolis criterion for each edge in the batch sequentially. This
        is repeated for a number of ``flip_ratio / batch_size * mesh.n_edges``
        times.

        Args:
            mesh (TriMesh): input mesh to be used
            estore (EnergyManager): instance of :class:`EnergyManager` used in
                combination with the ``mesh`` to evaluate energy differences
                necessary for flip acceptance/rejection.
            flip_ratio (float): ratio of edges to test (must be in [0,1]).
        )pbdoc"
    );
}

}
