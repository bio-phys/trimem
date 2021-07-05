/** \file energy.cpp
 * \brief Helfrich Energy functional on a OpenMesh::TriMesh.
 */
#include "energy.h"

#include "numpy_util.h"
#include "mesh_tether.h"
#include "kernel.h"

#include "pybind11/iostream.h"

namespace trimem {

EnergyManager::EnergyManager(const TriMesh* mesh,
                             const EnergyParams& energy_params,
                             const ContinuationParams& continuation_params) :
  mesh_(mesh),
  params(energy_params),
  cparams_(continuation_params)
{
    // setup bond potential
    BondParams& bparams = params.bond_params;
    if (bparams.type == BondType::Edge)
    {
        bonds_ = std::make_unique<FlatBottomEdgePenalty>(bparams);
    }
    else if (bparams.type == BondType::Area)
    {
        bonds_ = std::make_unique<HarmonicTriAreaPenalty>(bparams);
    }
    else
        throw std::runtime_error("Unknown bond potential");

    // evaluate initial properties
    auto dump = energy();

    // set target properties
    target_props_.area      = properties.area * cparams_.area_frac;
    target_props_.volume    = properties.volume * cparams_.volume_frac;
    target_props_.curvature = properties.curvature * cparams_.curvature_frac;

    // set initial properties
    initial_props_ = target_props_;

    // update reference properties
    if (cparams_.delta > 1.0)
        throw std::runtime_error("Use ref_delta in range [0,1]");
    if (cparams_.lambda > 1.0)
        throw std::runtime_error("Use ref_lambda in range [0,1]");
    update_reference_properties();

    init_ = true;
}

void EnergyManager::update_reference_properties()
{
    if (cparams_.lambda < 1.0)
    {
        cparams_.lambda += cparams_.delta;
        real& lambda = cparams_.lambda;

        params.ref_volume      = (1.0 - lambda) * initial_props_.volume +
                                 lambda * target_props_.volume;
        params.ref_area        = (1.0 - lambda) * initial_props_.area +
                                 lambda * target_props_.area;
        params.ref_curvature   = (1.0 - lambda) * initial_props_.curvature +
                                 lambda * target_props_.curvature;
    }
}

real EnergyManager::energy()
{
    TrimemEnergy kernel(params, *mesh_, *bonds_);

    VertexProperties props{ 0.0, 0.0, 0.0, 0.0, 0.0 };
    parallel_reduction(mesh_->n_vertices(), kernel, props);

    properties = props;
    return kernel.final(props);
}

std::vector<Point> EnergyManager::gradient()
{
    size_t n = mesh_->n_vertices();

    // update global properties
    auto dump = energy();

    // properties gradients
    VertexPropertiesGradient zeros
      { Point(0), Point(0), Point(0), Point(0), Point(0) };
    std::vector<VertexPropertiesGradient> gprops(n, zeros);
    TrimemPropsGradient pg_kernel(*mesh_, *bonds_, gprops);
    parallel_for(n, pg_kernel);

    // evaluate gradient
    std::vector<Point> gradient(n, Point(0));
    TrimemGradient g_kernel(params, properties, gprops, gradient);
    parallel_for(n, g_kernel);

    return gradient;
}

std::ostream& operator<<(std::ostream& out, const EnergyManager& lhs)
{
  const EnergyParams& params = lhs.params;
  const VertexProperties& props = lhs.properties;

  out << "----- EnergyManager info\n";
  out << "reference properties:\n";
  out << "  area:      " << params.ref_area << "\n";
  out << "  volume:    " << params.ref_volume << "\n";
  out << "  curvature: " << params.ref_curvature << "\n";
  out << "current properties:\n";
  out << "  area:      " << props.area << "\n";
  out << "  volume:    " << props.volume << "\n";
  out << "  curvature: " << props.curvature << "\n";
  out << "energies:\n";
  out << "  area:      " << area_penalty(params, props) << "\n";
  out << "  volume:    " << volume_penalty(params, props) << "\n";
  out << "  area diff: " << curvature_penalty(params, props) << "\n";
  out << "  bending:   " << helfrich_energy(params, props) << "\n";
  out << "  tether:    " << tether_potential(params, props) << "\n";
  out << "  total:     " << trimem_energy(params, props) << "\n";
  out << std::endl;

  return out;
}

void EnergyManager::print_info()
{
    std::cout << *this;
}

void expose_energy(py::module& m){

    py::class_<EnergyManager>(m, "EnergyManager")
       .def(py::init<TriMesh*, EnergyParams, ContinuationParams>())
       .def("energy", &EnergyManager::energy)
       .def("gradient", [](EnergyManager& _self){
          auto grad = _self.gradient();
          return tonumpy(grad[0], grad.size());})
       .def("update_reference_properties",
            &EnergyManager::update_reference_properties)
       .def("print_info",
            &EnergyManager::print_info,
            py::call_guard<py::scoped_ostream_redirect,
            py::scoped_estream_redirect>())
       .def_readwrite("properties", &EnergyManager::properties);
}

}
