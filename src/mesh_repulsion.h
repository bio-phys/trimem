/** \file mesh_repulsion.h
 * \brief Penalty on mesh self-intersection using a cell-list or verlet-list.
 */
#ifndef SURFACE_REPULSION_H
#define SURFACE_REPULSION_H

#include "defs.h"

#include "cell_list.h"
#include "neighbour_list.h"
#include "params.h"

namespace trimem {

//! Interface to SurfaceRepulsion
struct SurfaceRepulsion
{
    //! Compute penalty contributions to VertexProperties
    virtual real vertex_property(const TriMesh &mesh, const int &pid) const = 0;

    //! compute gradient of contributions to VertexProperties
    virtual std::tuple<std::vector<Point>, std::vector<int>>
    vertex_property_grad(const TriMesh &mesh, const int &pid) const = 0;
};

//! SurfaceRepulsion with template on neighbour search.
// TODO: assert valid template types at compile time
template <class ListType>
struct SurfaceRepulsionT : SurfaceRepulsion
{
    ListType nlist;

    SurfaceRepulsionT(const TriMesh &mesh, const double &rlist)
        : nlist(mesh, rlist) {}

    virtual real vertex_property(const TriMesh &mesh, const int &pid) const = 0;

    virtual std::tuple<std::vector<Point>, std::vector<int>>
    vertex_property_grad(const TriMesh &mesh, const int &pid) const = 0;
};

//!  SurfaceRepulsion with flat bottom potential.
template <class ListType>
struct SurfaceRepulsionFlatBottom : SurfaceRepulsionT<ListType>
{
    int r_;
    real lc1_;

    SurfaceRepulsionFlatBottom(const TriMesh &mesh,
                               const SurfaceRepulsionParams &params)
        : SurfaceRepulsionT<ListType>(mesh, params.rlist),
          r_(params.r), lc1_(params.lc1) {}

    virtual real
    vertex_property(const TriMesh& mesh, const int& pid) const override
    {
        // distances of vertex i (vid) to every other vertex j
        std::vector<Point> dij;
        std::vector<int> jdx;
        std::tie(dij,jdx) = this->nlist.point_distances(mesh, pid, lc1_);

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
        std::tie(dij, jdx) = this->nlist.point_distances(mesh, pid, lc1_);

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
make_repulsion(const TriMesh& mesh, const SurfaceRepulsionParams& params)
{
    // neighbour search with exclusions
    typedef CellList<1> CL1;
    typedef CellList<2> CL2;
    typedef NeighbourLists<1> NL1;
    typedef NeighbourLists<2> NL2;

    if (params.n_search == "cell-list")
    {
      if (params.exclusion_level == 1)
      {
        return std::make_unique<SurfaceRepulsionFlatBottom<CL1>>(mesh, params);
      }
      else if (params.exclusion_level == 2)
      {
        return std::make_unique<SurfaceRepulsionFlatBottom<CL2>>(mesh, params);
      }
      else
        throw std::runtime_error("Unsupported exclusion level");
    }
    else if (params.n_search == "verlet-list")
    {
      if (params.exclusion_level == 1)
      {
        return std::make_unique<SurfaceRepulsionFlatBottom<NL1>>(mesh, params);
      }
      else if (params.exclusion_level == 2)
      {
        return std::make_unique<SurfaceRepulsionFlatBottom<NL2>>(mesh, params);
      }
      else
        throw std::runtime_error("Unsupported exclusion level");
    }
    else
        throw std::runtime_error("Unknown neighbour search algorithm.");
};

}
#endif
