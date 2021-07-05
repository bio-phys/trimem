/** \file kernel.h
 * \brief Compute-kernels for the helfrich energy.
 */
#ifndef KERNEL_H
#define KERNEL_H
#include <memory>

#include "defs.h"

#include "params.h"
#include "mesh_properties.h"
#include "mesh_tether.h"

#include "pybind11/pybind11.h"

namespace trimem {

//! energy contributions
real area_penalty(const EnergyParams& params, const VertexProperties& props)
{
    real d = props.area / params.ref_area - 1.0;
    return params.kappa_a * d * d;
}

Point area_penalty_grad(const EnergyParams& params,
                        const VertexProperties& props,
                        const Point& d_area)
{
    real d   = props.area / params.ref_area - 1.0;
    real fac = 2.0 * params.kappa_a / params.ref_area * d;
    return fac * d_area;
}

real volume_penalty(const EnergyParams& params, const VertexProperties& props)
{
    real d = props.volume / params.ref_volume - 1.0;
    return params.kappa_v * d * d;
}

Point volume_penalty_grad(const EnergyParams& params,
                          const VertexProperties& props,
                          const Point& d_volume)
{
    real d = props.volume / params.ref_volume - 1.0;
    real fac = 2.0 * params.kappa_v / params.ref_volume * d;
    return fac * d_volume;
}

real curvature_penalty(const EnergyParams& params,
                       const VertexProperties& props)
{
    real d = props.curvature / params.ref_curvature - 1.0;
    return params.kappa_c * d * d;
}

Point curvature_penalty_grad(const EnergyParams& params,
                             const VertexProperties& props,
                             const Point& d_curvature)
{
    real d = props.curvature / params.ref_curvature - 1.0;
    real fac = 2.0 * params.kappa_c / params.ref_curvature * d;
    return fac * d_curvature;
}

real tether_potential(const EnergyParams& params, const VertexProperties& props)
{
    return params.kappa_t * props.tethering;
}

Point tether_potential_grad(const EnergyParams& params,
                            const VertexProperties& props,
                            const Point& d_tether)
{
    return params.kappa_t * d_tether;
}

real helfrich_energy(const EnergyParams& params, const VertexProperties& props)
{
    return params.kappa_b * props.bending;
}

Point helfrich_energy_grad(const EnergyParams& params,
                           const VertexProperties& props,
                           const Point& d_bending)
{
    return params.kappa_b * d_bending;
}

real trimem_energy(const EnergyParams& params, const VertexProperties& props)
{
    real energy = 0.0;
    energy += area_penalty(params, props);
    energy += volume_penalty(params, props);
    energy += curvature_penalty(params, props);
    energy += tether_potential(params, props);
    energy += helfrich_energy(params, props);
    return energy;
}

Point trimem_gradient(const EnergyParams& params,
                      const VertexProperties& props,
                      const VertexPropertiesGradient& gprops)
{
    Point grad(0.0);
    grad += area_penalty_grad(params, props, gprops.area);
    grad += volume_penalty_grad(params, props, gprops.volume);
    grad += curvature_penalty_grad(params, props, gprops.curvature);
    grad += tether_potential_grad(params, props, gprops.tethering);
    grad += helfrich_energy_grad(params, props, gprops.bending);
    return grad;
}

//! energy
struct TrimemEnergy
{
    TrimemEnergy(const EnergyParams& params,
                 const TriMesh& mesh,
                 const BondPotential& bonds) :
        params_(params),
        mesh_(mesh),
        bonds_(bonds) {}

    //parameters
    const EnergyParams& params_;
    const TriMesh& mesh_;
    const BondPotential& bonds_;

    void operator() (const int i, VertexProperties& contrib)
    {
        auto vh = mesh_.vertex_handle(i);
        contrib += vertex_properties(mesh_, bonds_, vh);
    }

    void operator() (const VertexHandle& vh, VertexProperties& contrib)
    {
        contrib += vertex_properties(mesh_, bonds_, vh);
    }

    real final(const VertexProperties& props)
    {
        return trimem_energy(params_, props);
    }
};

struct TrimemPropsGradient
{
    TrimemPropsGradient(const TriMesh& mesh,
                        const BondPotential& bonds,
                        std::vector<VertexPropertiesGradient>& gradients) :
        mesh_(mesh),
        bonds_(bonds),
        gradients_(gradients) {}

    //parameters
    const TriMesh& mesh_;
    const BondPotential& bonds_;

    // result
    std::vector<VertexPropertiesGradient>& gradients_;

    void operator() (const int i)
    {
        auto vh = mesh_.vertex_handle(i);
        vertex_properties_grad(mesh_, bonds_, vh, gradients_);
    }
};

struct TrimemGradient
{
    TrimemGradient(const EnergyParams& params,
                   const VertexProperties& props,
                   const std::vector<VertexPropertiesGradient>& gprops,
                   std::vector<Point>& gradient) :
        params_(params),
        props_(props),
        gprops_(gprops),
        gradient_(gradient) {}

    // parameters
    const EnergyParams& params_;
    const VertexProperties& props_;
    const std::vector<VertexPropertiesGradient>& gprops_;

    // result
    std::vector<Point>& gradient_;

    void operator() (const int i)
    {
        gradient_[i] += trimem_gradient(params_, props_, gprops_[i]);
    }

};

template<class Kernel, class ReductionType>
void parallel_reduction(int n, Kernel& kernel, ReductionType& reduce)
{
#pragma omp declare reduction (tred : ReductionType : omp_out += omp_in) \
  initializer(omp_priv={})

#pragma omp parallel for reduction(tred:reduce)
    for (int i=0; i<n; i++)
    {
        kernel(i, reduce);
    }
}

template<class Kernel>
void parallel_for(int n, Kernel& kernel)
{
#pragma omp parallel for
    for (int i=0; i<n; i++)
      kernel(i);
}

}
#endif
