/** \file mesh_repulsion.h
 * \brief Penalty on mesh self-intersection using a cell-list or verlet-list.
 */
#ifndef SURFACE_REPULSION_H
#define SURFACE_REPULSION_H

#include "defs.h"

#include "nlists/nlist.h"
#include "params.h"

namespace trimem {

//! Interface to SurfaceRepulsion
struct SurfaceRepulsion
{
    virtual ~SurfaceRepulsion() = default;

    //! Compute penalty contributions to VertexProperties
    virtual real vertex_property(const TriMesh &mesh, const int &pid) const = 0;

    //! compute gradient of contributions to VertexProperties
    virtual std::tuple<std::vector<Point>, std::vector<int>>
    vertex_property_grad(const TriMesh &mesh, const int &pid) const = 0;
};

//!  SurfaceRepulsion with flat bottom potential.
struct SurfaceRepulsionFlatBottom : SurfaceRepulsion
{
    const int           r_;
    const real          lc1_;
    const NeighbourList& nlist_;

    SurfaceRepulsionFlatBottom(const NeighbourList& nlist,
                               const SurfaceRepulsionParams &params) :
          r_(params.r), lc1_(params.lc1), nlist_(nlist) {}

    virtual real
    vertex_property(const TriMesh& mesh, const int& pid) const override
    {
        // distances of vertex i (vid) to every other vertex j
        std::vector<Point> dij;
        std::vector<int> jdx;
        std::tie(dij,jdx) = nlist_.point_distances(mesh, pid, lc1_);

        // penality potential for vertex vid
        real pot = 0;
        for (auto& d : dij)
        {
            real l = d.norm();
            pot += std::exp( l / ( l - lc1_ ) ) / std::pow( l, r_ );
        }
        return pot;
    }

    virtual std::tuple<std::vector<Point>, std::vector<int> >
    vertex_property_grad(const TriMesh& mesh, const int& pid) const override
    {
        // vertex_i - vertex_j
        std::vector<Point> dij;
        // vertex_j indices
        std::vector<int> jdx;
        std::tie(dij, jdx) = nlist_.point_distances(mesh, pid, lc1_);

        for (auto &d : dij)
        {
          real fac = 0.0;
          real l   = d.norm();
          real lr  = std::pow(l, r_+1);
          real lmc =  l - lc1_;
          fac -= lc1_ / ( lmc * lmc ) + r_ / l;
          fac /= lr;
          fac *= std::exp( l / lmc );
          d *= fac;
        }

        return std::make_tuple(dij, jdx);
    }
};

inline std::unique_ptr<SurfaceRepulsion>
make_repulsion(const NeighbourList& nlist, const SurfaceRepulsionParams& params)
{
    return std::make_unique<SurfaceRepulsionFlatBottom>(nlist, params);
};

}
#endif
