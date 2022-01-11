/** \file energy.cpp
 * \brief Helfrich Energy functional on a OpenMesh::TriMesh.
 */
#include "energy.h"

#include "numpy_util.h"
#include "mesh_tether.h"
#include "mesh_repulsion.h"
#include "kernel.h"

#include "pybind11/iostream.h"

namespace trimem {

EnergyManager::EnergyManager(const TriMesh* mesh,
                             const EnergyParams& energy_params) :
  mesh_(mesh),
  params(energy_params)
{
    // setup bond potential
    bonds = make_bonds(params.bond_params);

    // setup mesh repulsion
    update_repulsion();

    // evaluate properties from mesh
    auto dump = energy();

    // set initial properties
    initial_props = properties;

    // set reference properties
    interpolate_reference_properties();
}

void EnergyManager::set_mesh(const TriMesh* mesh)
{
    mesh_ = mesh;
}

void EnergyManager::interpolate_reference_properties()
{
    auto& cparams = params.continuation_params;

    real i_af = 1.0 - params.area_frac;
    real i_vf = 1.0 - params.volume_frac;
    real i_cf = 1.0 - params.curvature_frac;
    real& lam = cparams.lambda;

    ref_props.area      = ( 1.0 - lam * i_af ) * initial_props.area;
    ref_props.volume    = ( 1.0 - lam * i_vf ) * initial_props.volume;
    ref_props.curvature = ( 1.0 - lam * i_cf ) * initial_props.curvature;
}

void EnergyManager::update_reference_properties()
{
    auto& cparams = params.continuation_params;

    if (cparams.lambda < 1.0)
    {
        cparams.lambda += cparams.delta;
    }
    interpolate_reference_properties();
}

void EnergyManager::update_repulsion()
{
    repulse = make_repulsion(*mesh_, params.repulse_params);
}

real EnergyManager::energy()
{
    TrimemEnergy kernel(params, *mesh_, *bonds, *repulse, ref_props);

    VertexProperties props{ 0, 0, 0, 0, 0, 0};
    parallel_reduction(mesh_->n_vertices(), kernel, props);

    properties = props;
    return kernel.final(props);
}

real EnergyManager::energy(VertexProperties& props)
{
    TrimemEnergy kernel(params, *mesh_, *bonds, *repulse, ref_props);
    return kernel.final(props);
}

std::vector<Point> EnergyManager::gradient()
{
    size_t n = mesh_->n_vertices();

    // update global properties
    auto dump = energy();

    // properties gradients
    VertexPropertiesGradient zeros
      { Point(0), Point(0), Point(0), Point(0), Point(0), Point(0) };
    std::vector<VertexPropertiesGradient> gprops(n, zeros);
    TrimemPropsGradient pg_kernel(*mesh_, *bonds, *repulse, gprops);
    parallel_for(n, pg_kernel);

    // evaluate gradient
    std::vector<Point> gradient(n, Point(0));
    TrimemGradient g_kernel(params, properties, ref_props, gprops, gradient);
    parallel_for(n, g_kernel);

    return gradient;
}

std::ostream& operator<<(std::ostream& out, const EnergyManager& lhs)
{
  const EnergyParams& params = lhs.params;
  const VertexProperties& props = lhs.properties;
  const VertexProperties& ref_props = lhs.ref_props;

  out << "----- EnergyManager info\n";
  out << "reference properties:\n";
  out << "  area:      " << ref_props.area << "\n";
  out << "  volume:    " << ref_props.volume << "\n";
  out << "  curvature: " << ref_props.curvature << "\n";
  out << "current properties:\n";
  out << "  area:      " << props.area << "\n";
  out << "  volume:    " << props.volume << "\n";
  out << "  curvature: " << props.curvature << "\n";
  out << "energies:\n";
  out << "  area:      " << area_penalty(params, props, ref_props) << "\n";
  out << "  volume:    " << volume_penalty(params, props, ref_props) << "\n";
  out << "  area diff: " << curvature_penalty(params, props, ref_props) << "\n";
  out << "  bending:   " << helfrich_energy(params, props) << "\n";
  out << "  tether:    " << tether_penalty(params, props) << "\n";
  out << "  repulsion: " << repulsion_penalty(params, props) << "\n";
  out << "  total:     " << trimem_energy(params, props, ref_props) << "\n";
  out << std::endl;

  return out;
}

void EnergyManager::print_info()
{
    std::cout << *this;
}

void expose_energy(py::module& m){

    py::class_<EnergyManager>(m, "EnergyManager")
        .def(py::init<TriMesh*, EnergyParams>())
        .def("set_mesh", &EnergyManager::set_mesh)
        .def("energy", static_cast<real (EnergyManager::*)()>
                (&EnergyManager::energy))
        .def("energy", static_cast<real (EnergyManager::*)(VertexProperties&)>
                (&EnergyManager::energy))
        .def("gradient", [](EnergyManager& _self){
            auto grad = _self.gradient();
            return tonumpy(grad[0], grad.size());})
        .def("update_reference_properties",
            &EnergyManager::update_reference_properties)
        .def("update_repulsion", &EnergyManager::update_repulsion)
        .def("print_info",
            &EnergyManager::print_info,
            py::call_guard<py::scoped_ostream_redirect,
            py::scoped_estream_redirect>())
        .def_readwrite("properties", &EnergyManager::properties)
        .def_readonly("eparams", &EnergyManager::params)
        .def_readonly("initial_props", &EnergyManager::initial_props)
        .def_readonly("ref_props", &EnergyManager::ref_props);

}

}
