/** \file energy.cpp
 * \brief Helfrich Energy functional on a OpenMesh::TriMesh.
 */
#include "energy.h"

#include "mesh_tether.h"
#include "nlists/nlist.h"
#include "mesh_repulsion.h"
#include "external.h"
#include "kernel.h"

namespace trimem {

EnergyManager::EnergyManager(const TriMesh& mesh,
                             const EnergyParams& energy_params) :
  params(energy_params)
{
    // setup bond potential
    bonds = make_bonds(params.bond_params);

    // setup neighbour list
    nlist = make_nlist(mesh, params);

    // setup mesh repulsion
    repulse = make_repulsion(*nlist, params.repulse_params);

    // external potential
    external = make_external(params.external_params);

    // evaluate properties from mesh
    initial_props = properties(mesh);
}

void EnergyManager::update()
{
    params.area_frac.update();
    params.volume_frac.update();
    params.curvature_frac.update();
    params.external_params.radius.update();
}

void EnergyManager::update_repulsion(const TriMesh& mesh)
{
    nlist   = make_nlist(mesh, params);
    repulse = make_repulsion(*nlist, params.repulse_params);
}

VertexProperties EnergyManager::properties(const TriMesh& mesh)
{
    const size_t n = mesh.n_vertices();

    VertexProperties props{ 0, 0, 0, 0, 0, 0};
    std::vector<VertexProperties> vprops(n, props);

    EvaluateProperties eval_kernel(params, mesh, *bonds, *repulse, *external, vprops);
    parallel_for(n, eval_kernel);

    ReduceProperties reduce_kernel(vprops);
    parallel_reduction(n, reduce_kernel, props);

    return props;
}

real EnergyManager::energy(const TriMesh& mesh)
{
    auto props = properties(mesh);

    return trimem_energy(params, props, initial_props);
}

real EnergyManager::energy(const VertexProperties& props)
{
    return trimem_energy(params, props, initial_props);
}

std::vector<Point> EnergyManager::gradient(const TriMesh& mesh)
{
    const size_t n = mesh.n_vertices();

    // update properties
    VertexProperties props{ 0, 0, 0, 0, 0, 0};
    std::vector<VertexProperties> vprops(n, props);

    EvaluateProperties eval_kernel(params, mesh, *bonds, *repulse, *external, vprops);
    parallel_for(n, eval_kernel);

    ReduceProperties reduce_kernel(vprops);
    parallel_reduction(n, reduce_kernel, props);

    // properties gradients
    VertexPropertiesGradient zeros
      { Point(0), Point(0), Point(0), Point(0), Point(0), Point(0) };
    std::vector<VertexPropertiesGradient> gprops(n, zeros);
    EvaluatePropertiesGradient pg_kernel(
        mesh, *bonds, *repulse, *external, vprops, gprops);
    parallel_for(n, pg_kernel);

    // evaluate gradient
    std::vector<Point> gradient(n, Point(0));
    EvaluateGradient g_kernel(params, props, initial_props, gprops, gradient);
    parallel_for(n, g_kernel);

    return gradient;
}

void EnergyManager::print_info(const TriMesh& mesh)
{
  auto props     = properties(mesh);

  auto ref_area = params.area_frac * initial_props.area;
  auto ref_volume = params.volume_frac * initial_props.volume;
  auto ref_curvature = params.curvature_frac * initial_props.curvature;

  std::ostream& out = std::cout;

  out << "----- EnergyManager info\n";
  out << "reference properties:\n";
  out << "  area:      " << ref_area << "\n";
  out << "  volume:    " << ref_volume << "\n";
  out << "  curvature: " << ref_curvature << "\n";
  out << "current properties:\n";
  out << "  area:      " << props.area << "\n";
  out << "  volume:    " << props.volume << "\n";
  out << "  curvature: " << props.curvature << "\n";
  out << "energies:\n";
  out << "  area:      " << area_penalty(params, props, initial_props) << "\n";
  out << "  volume:    " << volume_penalty(params, props, initial_props) << "\n";
  out << "  area diff: " << curvature_penalty(params, props, initial_props) << "\n";
  out << "  bending:   " << helfrich_energy(params, props) << "\n";
  out << "  tether:    " << tether_penalty(params, props) << "\n";
  out << "  repulsion: " << repulsion_penalty(params, props) << "\n";
  out << "  total:     " << trimem_energy(params, props, initial_props) << "\n";
  out << std::endl;
}


}
