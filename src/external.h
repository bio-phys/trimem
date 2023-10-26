/** \file external.h
 * \brief External local potentials acting on vertices.
 */
#ifndef EXTERNAL_H
#define EXTERNAL_H

#include "defs.h"

#include "params.h"

#include <autodiff/forward/real.hpp>
#include <autodiff/forward/real/eigen.hpp>

namespace trimem {

//! Interface to ExternalPotential
struct ExternalPotential
{
    virtual ~ExternalPotential() = default;

    //! Compute penalty contributions to VertexProperties
    virtual real vertex_property(const TriMesh &mesh, const VertexHandle& vh) const = 0;

    //! compute gradient of contributions to VertexProperties
    virtual Point
    vertex_property_grad(const TriMesh &mesh, const VertexHandle& vh) const = 0;
};

//! Do-nothing potential since external potentials are not standard
struct None : ExternalPotential
{
    None(const ExternalPotentialParams &params) {}
    virtual real
    vertex_property(const TriMesh& mesh, const VertexHandle& vh) const override
    {
        return 0;
    }

    virtual Point
    vertex_property_grad(const TriMesh& mesh, const VertexHandle& vh) const override
    {
        return Point(0);
    }
};

//! repulsive sphere
struct Sphere : ExternalPotential
{
    // parameters
    const ExternalPotentialParams& params_;
    static constexpr real twopow16 = std::pow(2, 1.0/6);

    struct SphereParams
    {
        autodiff::real epsilon;
        autodiff::real sigma;
        autodiff::real radius;
    };

    Sphere(const ExternalPotentialParams &params) :
          params_(params) {}

    static autodiff::real f(const autodiff::Array3real& x, const SphereParams& p)
    {
        autodiff::real n = abs( p.radius - sqrt((x * x).sum()) );

        autodiff::real V = 0;
        if ( n < (twopow16 * p.sigma) )
        {
          autodiff::real sigma_r_inv = p.sigma / n;
          V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
          V *= 4 * p.epsilon;
          V += 1;
        }

        return V;
    }

    virtual real
    vertex_property(const TriMesh& mesh, const VertexHandle& vh) const override
    {
        // coordinates of vertex with handle vh
        const auto& point = mesh.point(vh);

        autodiff::Array3real x{ point[0], point[1], point[2] };

        // this is a little odd, but gets the current radius correctly!
        SphereParams p{params_.epsilon, params_.sigma, params_.radius};

        autodiff::real u = f(x, p);
        return u[0];
    }

    virtual Point
    vertex_property_grad(const TriMesh& mesh, const VertexHandle& vh) const override
    {
        // coordinates of vertex with handle vh
        const auto& point = mesh.point(vh);

        autodiff::Array3real x{ point[0], point[1], point[2] };

        // this is a little odd, but gets the current radius correctly!
        SphereParams p{params_.epsilon, params_.sigma, params_.radius};

        autodiff::real u;

        Eigen::VectorXd g = gradient(f, wrt(x), at(x, p), u);

        return Point(g[0], g[1], g[2]);
    }
};

inline std::unique_ptr<ExternalPotential>
make_external(const ExternalPotentialParams& params)
{
    auto& ext_type = params.type;
    if (ext_type == "sphere")
        return std::make_unique<Sphere>(params);
    else if (ext_type == "none")
        return std::make_unique<None>(params);
    else
        throw std::runtime_error("Unknown external potential.");

};

}
#endif
