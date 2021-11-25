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

bool test_guards(TriMesh& mesh, const int& idx, std::vector<OmpGuard>& locks)
{
    // check if edge is still available at this point
    bool locked = locks[idx].test();

    if (locked)
    {
        auto eh          = mesh.edge_handle(idx);
        auto patch       = flip_patch(mesh, eh);
        int  count_locks = 0;

        // remove edge itself from patch
        patch.erase(idx);

        // try to lock entire patch
        for (const int& i: patch)
        {
            if (locks[i].test())
            {
                count_locks++;
            }
            else
            {
                break;
            }
        }

        // unrool in case of some clashes with other patches
        if (count_locks != patch.size())
        {
            locks[idx].release();
            for (const int& i: patch)
            {
                if (count_locks != 0)
                {
                    locks[i].release();
                    count_locks--;
                }
                else
                {
                    break;
                }
            }
            return false;
        }
    }
    return locked;
}
}
