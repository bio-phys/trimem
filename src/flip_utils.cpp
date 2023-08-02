/** \file flip_utils.cpp
 * \brief Utility functions for performing edge flips on openmesh.
 */
#include <unordered_set>
#include <algorithm>
#include <chrono>
#include <random>
#include <cmath>

#include "omp.h"

#include "flip_utils.h"
#include "omp_guard.h"

namespace trimem {

typedef std::chrono::high_resolution_clock myclock;

VertexProperties edge_vertex_properties(TriMesh& mesh,
                                       const EdgeHandle& eh,
                                       const BondPotential& bonds,
                                       const SurfaceRepulsion& repulse)

{
    VertexProperties props{ 0.0, 0.0, 0.0, 0.0, 0.0 };

    for (int i=0; i<2; i++)
    {
        // vertex properties of the first face
        auto heh = mesh.halfedge_handle(eh, i);

        auto ve = mesh.to_vertex_handle(heh);
        props += vertex_properties(mesh, bonds, repulse, ve);

        auto next_heh = mesh.next_halfedge_handle(heh);
        ve = mesh.to_vertex_handle(next_heh);
        props += vertex_properties(mesh, bonds, repulse, ve);
    }

    return props;
}


VertexPropertiesNSR edge_vertex_properties_nsr(TriMesh& mesh,
                                       const EdgeHandle& eh,
                                       const BondPotential& bonds
                                       )

{
    VertexPropertiesNSR props{ 0.0, 0.0, 0.0, 0.0};

    for (int i=0; i<2; i++)
    {
        // vertex properties of the first face
        auto heh = mesh.halfedge_handle(eh, i);

        auto ve = mesh.to_vertex_handle(heh);
        props += vertex_properties_nsr(mesh, bonds, ve);

        auto next_heh = mesh.next_halfedge_handle(heh);
        ve = mesh.to_vertex_handle(next_heh);
        props += vertex_properties_nsr(mesh, bonds,  ve);
    }

    return props;
}


std::unordered_set<int> flip_patch(TriMesh& mesh, const EdgeHandle& eh)
{
    // get all edges blocked by a flip of eh
    std::unordered_set<int> patch;
    for (int i=0; i<2; i++)
    {
        auto heh  = mesh.halfedge_handle(eh, i);
        auto vh   = mesh.to_vertex_handle(heh);
        for (auto h_it=mesh.voh_iter(vh); h_it.is_valid(); ++h_it)
        {
            patch.insert(mesh.edge_handle(*h_it).idx());
            auto n_he = mesh.next_halfedge_handle(*h_it);
            patch.insert(mesh.edge_handle(n_he).idx());
        }

        heh = mesh.next_halfedge_handle(heh);
        vh  = mesh.to_vertex_handle(heh);
        for (auto h_it=mesh.voh_iter(vh); h_it.is_valid(); ++h_it)
        {
            patch.insert(mesh.edge_handle(*h_it).idx());
            auto n_he = mesh.next_halfedge_handle(*h_it);
            patch.insert(mesh.edge_handle(n_he).idx());
        }
    }
    return patch;
}

/** Test locking of edges
 *
 *  Acquire locks on the lock-vector for the patch associated to edge idx.
 *  In case of success, return a set of OmpGuards 'owning' the locks. The locks
 *  are realased as soon as the returned set goes out of scope.
 */
std::vector<OmpGuard>
test_patch(TriMesh& mesh, const int& idx, std::vector<omp_lock_t>& locks)
{
    // check if edge is still available at this point
    OmpGuard edge_guard(locks[idx]);

    std::vector<OmpGuard> patch_guard;
    if (edge_guard.test())
    {
        // get edge-patch
        auto eh    = mesh.edge_handle(idx);
        auto patch = flip_patch(mesh, eh);

        // vector of guards initialized with the edge itself
        patch_guard.reserve(patch.size());
        patch_guard.push_back(std::move(edge_guard));

        // try to lock the entire patch but remove the edge
        // itself from patch first since it is already locked
        patch.erase(idx);
        for (const int& i: patch)
        {
            OmpGuard i_guard(locks[i]);
            if (i_guard.test())
            {
                // keep guard alive if lock was successful
                patch_guard.push_back(std::move(i_guard));
            }
            else
            {
                // release all locks
                patch_guard.clear();
                break;
            }
        }

    }
    return patch_guard;
}
}
