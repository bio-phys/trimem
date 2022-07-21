/** \file kernel.h
 * \brief Compute-kernels for the helfrich energy.
 */
#ifndef KERNEL_H
#define KERNEL_H
#include <omp.h>
#include <memory>

#include "defs.h"

#include "params.h"
#include "mesh_properties.h"
#include "mesh_tether.h"
#include "mesh_repulsion.h"

namespace trimem {

//! energy contributions
real area_penalty(const EnergyParams& params,
                  const VertexProperties& props,
                  const VertexProperties& ref_props)
{
    real d = props.area / ref_props.area - 1.0;
    return params.kappa_a * d * d;
}

Point area_penalty_grad(const EnergyParams& params,
                        const VertexProperties& props,
                        const VertexProperties& ref_props,
                        const Point& d_area)
{
    real d   = props.area / ref_props.area - 1.0;
    real fac = 2.0 * params.kappa_a / ref_props.area * d;
    return fac * d_area;
}

real volume_penalty(const EnergyParams& params,
                    const VertexProperties& props,
                    const VertexProperties& ref_props)
{
    real d = props.volume / ref_props.volume - 1.0;
    return params.kappa_v * d * d;
}

Point volume_penalty_grad(const EnergyParams& params,
                          const VertexProperties& props,
                          const VertexProperties& ref_props,
                          const Point& d_volume)
{
    real d = props.volume / ref_props.volume - 1.0;
    real fac = 2.0 * params.kappa_v / ref_props.volume * d;
    return fac * d_volume;
}

real curvature_penalty(const EnergyParams& params,
                       const VertexProperties& props,
                       const VertexProperties& ref_props)
{
    real d = props.curvature / ref_props.curvature - 1.0;
    return params.kappa_c * d * d;
}

Point curvature_penalty_grad(const EnergyParams& params,
                             const VertexProperties& props,
                             const VertexProperties& ref_props,
                             const Point& d_curvature)
{
    real d = props.curvature / ref_props.curvature - 1.0;
    real fac = 2.0 * params.kappa_c / ref_props.curvature * d;
    return fac * d_curvature;
}

real tether_penalty(const EnergyParams& params, const VertexProperties& props)
{
    return params.kappa_t * props.tethering;
}

Point tether_penalty_grad(const EnergyParams& params,
                          const VertexProperties& props,
                          const Point& d_tether)
{
    return params.kappa_t * d_tether;
}

real repulsion_penalty(const EnergyParams& params,
                       const VertexProperties& props)
{
    return params.kappa_r * props.repulsion;
}

Point repulsion_penalty_grad(const EnergyParams& params,
                             const VertexProperties& props,
                             const Point& d_repulsion)
{
    return params.kappa_r * d_repulsion;
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

real trimem_energy(const EnergyParams& params,
                   const VertexProperties& props,
                   const VertexProperties& ref_props)
{
    real energy = 0.0;
    energy += area_penalty(params, props, ref_props);
    energy += volume_penalty(params, props, ref_props);
    energy += curvature_penalty(params, props, ref_props);
    energy += tether_penalty(params, props);
    energy += repulsion_penalty(params, props);
    energy += helfrich_energy(params, props);
    return energy;
}

Point trimem_gradient(const EnergyParams& params,
                      const VertexProperties& props,
                      const VertexProperties& ref_props,
                      const VertexPropertiesGradient& gprops)
{
    Point grad(0.0);
    grad += area_penalty_grad(params, props, ref_props, gprops.area);
    grad += volume_penalty_grad(params, props, ref_props,  gprops.volume);
    grad += curvature_penalty_grad(params, props, ref_props, gprops.curvature);
    grad += tether_penalty_grad(params, props, gprops.tethering);
    grad += repulsion_penalty_grad(params, props, gprops.repulsion);
    grad += helfrich_energy_grad(params, props, gprops.bending);
    return grad;
}

//! energy
struct TrimemProperties
{
    TrimemProperties(const EnergyParams& params,
                     const TriMesh& mesh,
                     const BondPotential& bonds,
                     const SurfaceRepulsion& repulse) :
        params_(params),
        mesh_(mesh),
        bonds_(bonds),
        repulse_(repulse) {}

    //parameters
    const EnergyParams& params_;
    const TriMesh& mesh_;
    const BondPotential& bonds_;
    const SurfaceRepulsion& repulse_;

    void operator() (const int i, VertexProperties& contrib)
    {
        auto vh = mesh_.vertex_handle(i);
        contrib += vertex_properties(mesh_, bonds_, repulse_, vh);
    }

    void operator() (const VertexHandle& vh, VertexProperties& contrib)
    {
        contrib += vertex_properties(mesh_, bonds_, repulse_, vh);
    }
};

struct TrimemPropsGradient
{
    TrimemPropsGradient(const TriMesh& mesh,
                        const BondPotential& bonds,
                        const SurfaceRepulsion& repulse,
                        std::vector<VertexPropertiesGradient>& gradients) :
        mesh_(mesh),
        bonds_(bonds),
        repulse_(repulse),
        gradients_(gradients) {}

    //parameters
    const TriMesh& mesh_;
    const BondPotential& bonds_;
    const SurfaceRepulsion& repulse_;

    // result
    std::vector<VertexPropertiesGradient>& gradients_;

    void operator() (const int i)
    {
        auto vh = mesh_.vertex_handle(i);
        vertex_properties_grad(mesh_, bonds_, repulse_, vh, gradients_);
    }
};

struct TrimemGradient
{
    TrimemGradient(const EnergyParams& params,
                   const VertexProperties& props,
                   const VertexProperties& ref_props,
                   const std::vector<VertexPropertiesGradient>& gprops,
                   std::vector<Point>& gradient) :
        params_(params),
        props_(props),
        ref_props_(ref_props),
        gprops_(gprops),
        gradient_(gradient) {}

    // parameters
    const EnergyParams& params_;
    const VertexProperties& props_;
    const VertexProperties& ref_props_;
    const std::vector<VertexPropertiesGradient>& gprops_;

    // result
    std::vector<Point>& gradient_;

    void operator() (const int i)
    {
        gradient_[i] += trimem_gradient(params_, props_, ref_props_, gprops_[i]);
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
