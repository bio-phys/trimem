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
    static constexpr real twopow16 = 1.122462048309373; //2^(1/6)

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

//! repulsive sphere plus hole
struct Sphere_Hole : ExternalPotential
{
    // parameters
    const ExternalPotentialParams& params_;
    static constexpr real twopow16 = std::pow(2, 1.0/6);

    struct SphereHoleParams
    {
        autodiff::real epsilon;
        autodiff::real sigma;
        autodiff::real radius; //R
        autodiff::real alpha; //This should be in radians
    };

    Sphere_Hole(const ExternalPotentialParams &params) :
          params_(params) {}

    static autodiff::real f(const autodiff::Array3real& x, const SphereHoleParams& p)
    {
        autodiff::real theta = acos(x[2]/(sqrt((x * x).sum()))); //CHECK should be z/r
        autodiff::real V = 0;

        if ( theta >= p.alpha ) // if theta is greater or equal to alpha the LJ is calculated
        {
            autodiff::real n = abs( p.radius - sqrt((x * x).sum()) );
            if ( n < (twopow16 * p.sigma) )
            {
            autodiff::real sigma_r_inv = p.sigma / n;
            V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
            V *= 4 * p.epsilon;
            V += p.epsilon;
            }
	else //Direct interaction with the pore rim
        {
            autodiff::real Rho_0 = p.radius * sin(p.alpha);
            autodiff::real Z0 = p.radius * cos(p.alpha);
            autodiff::real n = sqrt( (Rho_0 - sqrt(x[0]*x[0] + x[1]*x[1]))*(Rho_0 - sqrt(x[0]*x[0] + x[1]*x[1]))  + (Z0 - x[2])*(Z0 - x[2]) );
            if ( n < (twopow16 * p.sigma) )
            {
            autodiff::real sigma_r_inv = p.sigma / n;
            V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
            V *= 4 * p.epsilon;
            V += p.epsilon;
            }
        }

        /*        
        else //meaning that theta is less than alpha 
        {
            autodiff::real delta_r=sqrt((p.radius*p.radius)+((x * x).sum())-(2*p.radius*(sqrt((x * x).sum()))*cos(theta-p.alpha)));
            autodiff::real V = 0;
            if ( delta_r < (twopow16 * p.sigma) )
            {
            autodiff::real sigma_r_inv = p.sigma / delta_r;
            V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
            V *= 4 * p.epsilon;
            V += p.epsilon;
            }
        }*/
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
        SphereHoleParams p{params_.epsilon, params_.sigma, params_.radius, params_.alpha};

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
        SphereHoleParams p{params_.epsilon, params_.sigma, params_.radius, params_.alpha};

        autodiff::real u;

        Eigen::VectorXd g = gradient(f, wrt(x), at(x, p), u);

        return Point(g[0], g[1], g[2]);
    }
};

//! repulsive cylindrical body with two hemispherical cap
struct Cylinder_two_hemisphere : ExternalPotential
{
    // parameters
    const ExternalPotentialParams& params_;
    static constexpr real twopow16 = std::pow(2, 1.0/6);

    struct MITO_Params
    {
        autodiff::real epsilon;
        autodiff::real sigma;
        autodiff::real radius; //radius of the cylindrical body, which is identical to that of hemispheres
        autodiff::real height; //Here, alpha will be used as a height X 2 of the cylindrical body
    };

    Cylinder_two_hemisphere(const ExternalPotentialParams &params) :
          params_(params) {}

    static autodiff::real f(const autodiff::Array3real& x, const MITO_Params& p)
    {
        //autodiff::real theta = acos(x[2]/(sqrt((x * x).sum()))); //CHECK should be z/r
        autodiff::real V = 0;

        if ( abs ( x[2] )<=p.height ) // if the point is within the cylindrical body
        {
            autodiff::real n = abs( p.radius - sqrt(x[0]*x[0]+x[1]*x[1]) );
            if ( n < (twopow16 * p.sigma) )
            {
            autodiff::real sigma_r_inv = p.sigma / n;
            V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
            V *= 4 * p.epsilon;
            V += p.epsilon;
            }
        }
        else
        {
            autodiff::real n = abs( p.radius - sqrt( x[0]*x[0]+x[1]*x[1]+(abs(x[2])-p.height)*(abs(x[2])-p.height) ) );
            if ( n < (twopow16 * p.sigma) )
            {
            autodiff::real sigma_r_inv = p.sigma / n;
            V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
            V *= 4 * p.epsilon;
            V += p.epsilon;
            }
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
        MITO_Params p{params_.epsilon, params_.sigma, params_.radius, params_.height};

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
        MITO_Params p{params_.epsilon, params_.sigma, params_.radius, params_.height};

        autodiff::real u;

        Eigen::VectorXd g = gradient(f, wrt(x), at(x, p), u);

        return Point(g[0], g[1], g[2]);
    }
};

//! repulsive cylindrical body with two hemispherical cap with a pore at the z axis
struct Porous_mitochondria : ExternalPotential
{
    // parameters
    const ExternalPotentialParams& params_;
    static constexpr real twopow16 = std::pow(2, 1.0/6);

    struct porous_MITO_Params
    {
        autodiff::real epsilon;
        autodiff::real sigma;
        autodiff::real radius; //Radius of the cylindrical body, which is identical to that of hemispheres
        autodiff::real alpha; //This should be in radians
        autodiff::real height; //Height X 2 of the cylindrical body
    };

    Porous_mitochondria(const ExternalPotentialParams &params) :
          params_(params) {}

    static autodiff::real f(const autodiff::Array3real& x, const porous_MITO_Params& p)
    {
        autodiff::real V = 0;

        if ( abs ( x[2] )<=p.height ) // if the point is within the cylindrical body
        {
            autodiff::real n = abs( p.radius - sqrt(x[0]*x[0]+x[1]*x[1]) );
            if ( n < (twopow16 * p.sigma) )
            {
            autodiff::real sigma_r_inv = p.sigma / n;
            V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
            V *= 4 * p.epsilon;
            V += p.epsilon;
            }
        }
	else if ( x[2] < -p.height )
	{
	    autodiff::real n = abs( p.radius - sqrt( x[0]*x[0]+x[1]*x[1]+(abs(x[2])-p.height)*(abs(x[2])-p.height) ) );
	    if ( n < (twopow16 * p.sigma) )
            {
            autodiff::real sigma_r_inv = p.sigma / n;
            V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
            V *= 4 * p.epsilon;
            V += p.epsilon;
            }
	}
        else // ( x[2] > p.height )
	{
	    autodiff::real shifted_Z = abs(x[2] - p.height);
	    autodiff::real theta = acos( shifted_Z/(sqrt( x[0]*x[0]+x[1]*x[1]+shifted_Z*shifted_Z )));
	    if ( theta >= p.alpha )
            {
                autodiff::real n = abs( p.radius - sqrt( x[0]*x[0]+x[1]*x[1]+(abs(x[2])-p.height)*(abs(x[2])-p.height) ) );
                if ( n < (twopow16 * p.sigma) )
                {
                autodiff::real sigma_r_inv = p.sigma / n;
                V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
                V *= 4 * p.epsilon;
                V += p.epsilon;
                }
            }
            else //Direct interaction with the pore rim
            {
                autodiff::real Rho_0 = p.radius * sin(p.alpha);
                autodiff::real Z0 = p.radius * cos(p.alpha);
                autodiff::real n = sqrt( (Rho_0 - sqrt(x[0]*x[0] + x[1]*x[1]))*(Rho_0 - sqrt(x[0]*x[0] + x[1]*x[1]))  + (Z0 - shifted_Z)*(Z0 - shifted_Z) );
                if ( n < (twopow16 * p.sigma) )
                {
                autodiff::real sigma_r_inv = p.sigma / n;
                V  = pow( sigma_r_inv , 12 ) - pow( sigma_r_inv, 6 );
                V *= 4 * p.epsilon;
                V += p.epsilon;
                }
            }
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
        porous_MITO_Params p{params_.epsilon, params_.sigma, params_.radius, params_.alpha, params_.height};

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
        porous_MITO_Params p{params_.epsilon, params_.sigma, params_.radius, params_.alpha, params_.height};

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
    else if (ext_type == "spherephole")
        return std::make_unique<Sphere_Hole>(params);
    else if (ext_type == "MITO")
        return std::make_unique<Cylinder_two_hemisphere>(params);
    else if (ext_type == "porous_MITO")
        return std::make_unique<Porous_mitochondria>(params);
    else if (ext_type == "none")
        return std::make_unique<None>(params);
    else
        throw std::runtime_error("Unknown external potential.");

};

}
#endif
