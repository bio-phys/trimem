/** \file verlet_list.h
 * \brief Verlet list to be used with openmesh.
 */
#ifndef VERLET_LIST_H
#define VERLET_LIST_H

#include <vector>
#include <map>

#include "MeshTypes.hh"

#include "cell_list.h"

namespace trimem {

template<int exclusion = 0>
struct VerletList
{
    std::vector<std::vector<int> >neighbours;

    VerletList(const TriMesh& mesh, double rlist, double box_eps=1.e-6)
    {
        CellList<exclusion> clist(mesh, rlist, box_eps);

        // get indices of vertices in the rlist-ball
        auto res = clist.distance_matrix(mesh, rlist);
        auto& idx = std::get<1>(res);
        auto& jdx = std::get<2>(res);

        // loop over entries in idx
        neighbours.resize(mesh.n_vertices());
        auto jt = jdx.begin();
        for (auto it=idx.begin(); it!=idx.end(); ++it, ++jt)
        {
// TODO: the following should actually be superfluous!
/*
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
*/
            neighbours[*it].push_back(*jt);
            neighbours[*jt].push_back(*it);
        }
    }

    std::tuple<std::vector<double>, std::vector<int>, std::vector<int> >
    distance_matrix(const TriMesh& mesh, const double& dmax) const
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
                double dist = distance(idata, odata, 3);
                if (dist < dmax)
                {
                    dists.push_back(dist);
                    idx.push_back(i);
                    jdx.push_back(*it);
                }
            }
        }
        return std::make_tuple(std::move(dists),std::move(idx), std::move(jdx));
    }

    int distance_counts(const TriMesh& mesh, const double& dmax) const
    {
        double dmax2 = dmax * dmax;

        const TriMesh::Point& point = mesh.point(mesh.vertex_handle(0));
        const double *data = point.data();

        int ni=0;

        //loop over all vertices in the list
        #pragma omp parallel for reduction(+:ni)
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
                double dist = squ_distance(idata, odata, 3);
                if (dist < dmax2)
                {
                    ni += 1;
                }
            }
        }

        return ni;
    }

    std::tuple<std::vector<Point>, std::vector<int> >
    point_distances(const TriMesh& mesh, const int& pid, const double& dmax) const
    {
        std::vector<Point> dist;
        std::vector<int>   jdx;

        // check whether there are any neighbours in the list for this point
        auto& neighs = neighbours.at(pid);
        if (neighs.size() == 0)
        {
            return std::make_tuple(dist, neighs);
        }

        const auto& point = mesh.point(mesh.vertex_handle(pid));

        for (auto jid: neighs)
        {
            Point di = point - mesh.point(mesh.vertex_handle(jid));
            if (di.norm() < dmax)
            {
                dist.push_back(di);
                jdx.push_back(jid);
            }
        }

        return std::make_tuple(std::move(dist), std::move(jdx));
    }

    int point_distance_counts(const TriMesh& mesh,
                              const int& pid,
                              const double& dmax) const
    {
        auto res = point_distances(mesh, pid, dmax);
        return std::get<0>(res).size();
    }

};

}
#endif
