/** \file mesh_tether.h
 * \brief Tethering constraints on a OpenMesh::TriMesh.
 */
#ifndef MESH_TETHER_H
#define MESH_TETHER_H

#include "defs.h"
#include "params.h"
#include "mesh_util.h"

namespace trimem {

struct BondPotential
{
    virtual ~BondPotential() = default;

    virtual real vertex_property(const TriMesh& mesh,
                                 const HalfedgeHandle& he) const = 0;

    virtual std::vector<Point>
    vertex_property_grad(const TriMesh& mesh,
                         const HalfedgeHandle& he) const = 0;

    virtual int valence() const = 0;
};

struct FlatBottomEdgePenalty : BondPotential
{
    FlatBottomEdgePenalty(const BondParams& params) :
      lc0_(params.lc0),
      lc1_(params.lc1),
      r_(params.r) {}

    real lc0_;
    real lc1_;
    int  r_;

    virtual real vertex_property(const TriMesh& mesh,
                                 const HalfedgeHandle& he) const override
    {
        real l = edge_length(mesh, he);
        if (l > lc0_)
        {
            return std::pow( r_, r_ + 1 ) * std::pow( l - lc0_, r_ );
        }
        if (l < lc1_)
        {
            return std::exp( l / ( l - lc1_ ) ) / std::pow( l, r_ );
        }
        else
          return 0.0;
    }

    virtual std::vector<Point>
    vertex_property_grad(const TriMesh& mesh,
                         const HalfedgeHandle& he) const override
    {
        real l  = edge_length(mesh, he);
        auto lg = edge_length_grad<3>(mesh, he);
        real fac = 0.0;
        if (l > lc0_)
        {
            fac = std::pow( r_, r_ + 2 ) * std::pow( l - lc0_, r_ - 1);
        }
        if (l < lc1_)
        {
            real lr  = std::pow(l, r_);
            real lmc =  l - lc1_;
            fac -= lc1_ / ( lmc * lmc ) + r_ / l;
            fac /= lr;
            fac *= std::exp( l / lmc );
        }

        for (auto& lgi : lg)
            lgi *= fac;

        return lg;
    }

    virtual int valence() const override { return 2; }
};

struct HarmonicTriAreaPenalty : BondPotential
{
    HarmonicTriAreaPenalty(const BondParams& params) :
        a0_(params.a0) {}

    real a0_;

    virtual real vertex_property(const TriMesh& mesh,
                                 const HalfedgeHandle& he) const override
    {
        real a = face_area(mesh, he);
        real d = a / a0_ - 1.0;
        return d * d;
    }

    virtual std::vector<Point>
    vertex_property_grad(const TriMesh& mesh,
                         const HalfedgeHandle& he) const override
    {
        real a  = face_area(mesh, he);
        auto ag = face_area_grad<7>(mesh, he);

        real d = 2.0 * ( a / a0_ - 1.0 ) / a0_;

        for (auto& agi : ag)
            agi *= d;

        return ag;
    }

    virtual int valence() const override { return 3; }

};

inline std::unique_ptr<BondPotential> make_bonds(const BondParams& params)
{
    if (params.type == BondType::Edge)
    {
        return std::make_unique<FlatBottomEdgePenalty>(params);
    }
    else if (params.type == BondType::Area)
    {
        return std::make_unique<HarmonicTriAreaPenalty>(params);
    }
    else
        throw std::runtime_error("Unknown bond potential");
};

}
#endif
