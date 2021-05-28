/** \file neighbours.h
 * \brief Neighbour list tools to be used with openmesh
 */
#ifndef NEIGHBOURS_H
#define NEIGHBOURS_H

#include <vector>
#include <map>

#include "MeshTypes.hh"

#include "cell_list.h"

namespace trimem {

template<bool exclude_self=true, bool exclude_one_ring=true>
struct NeighbourLists
{
    std::map<int, std::vector<int> > neighbours;

    NeighbourLists(const TriMesh& mesh, double rlist, double box_eps=1.e-6)
    {
        CellList clist(mesh, rlist, box_eps);

        // get indices of vertices in the rlist-ball
        auto res = clist.distance_matrix<exclude_self, exclude_one_ring>(
                      mesh, rlist);
        auto& idx = std::get<1>(res);
        auto& jdx = std::get<2>(res);

        // loop over entries in idx
        auto jt = jdx.begin();
        for (auto it=idx.begin(); it!=idx.end(); ++it, ++jt)
        {
            if (exclude_self)
            {
                if (*it==*jt) continue;
            }
            if (exclude_one_ring)
            {
                bool in_ring = false;
                auto vh = mesh.vertex_handle(*it);
                for (auto vit=mesh.cvv_iter(vh); vit.is_valid(); vit++)
                {
                    if (vit->idx() == *jt)
                    {
                        in_ring = true;
                        break;
                    }
                }
                if (in_ring) continue;
            }

            neighbours[*it].push_back(*jt);
            neighbours[*jt].push_back(*it);
        }
    }

    std::tuple<std::vector<double>, std::vector<int>, std::vector<int> >
    distance_matrix(const TriMesh& mesh, const double& rlist)
    {
        const TriMesh::Point& point = mesh.point(mesh.vertex_handle(0));
        const double *data = point.data();

        // sparse matrix data
        std::vector<double> dists;
        std::vector<int>    idx;
        std::vector<int>    jdx;

        //loop over all vertices in the list
        for (int i=0; i<mesh.n_vertices(); i++)
        {
            // this vertex's coordinates
            const double* idata = data+i*3;

            auto& neighs = neighbours.at(i);
            for (auto it=neighs.begin(); it!=neighs.end(); ++it)
            {
                // other vertex's coordinates
                const double* odata = data+*it*3;

                // compute distance and count in case
                double dist = distance<double>(idata, odata, 3);
                if (dist <= rlist)
                {
                    dists.push_back(dist);
                    idx.push_back(i);
                    jdx.push_back(*it);
                }
            }
        }
        return std::make_tuple(std::move(dists),std::move(idx), std::move(jdx));
    }

    int distance_counts(const TriMesh& mesh, const double& dmax)
    {
        const TriMesh::Point& point = mesh.point(mesh.vertex_handle(0));
        const double *data = point.data();

        int ni=0;

        //loop over all vertices in the list
        for (int i=0; i<mesh.n_vertices(); i++)
        {
            // this vertex's coordinates
            const double* idata = data+i*3;

            auto& neighs = neighbours.at(i);
            for (auto it=neighs.begin(); it!=neighs.end(); ++it)
            {
                // other vertex's coordinates
                const double* odata = data+*it*3;

                // compute distance and count in case
                double dist = distance<double>(idata, odata, 3);
                if (dist <= dmax)
                {
                    ni += 1;
                }
            }
        }

        return ni;
    }
};

}
#endif
