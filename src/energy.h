/** \file energy.h
 * \brief Helfrich Energy functional on a OpenMesh::TriMesh.
 */
#ifndef ENERGY_H
#define ENERGY_H
#include <memory>

#include "defs.h"
#include "params.h"
#include "mesh_properties.h"

#include "pybind11/pybind11.h"

namespace trimem {

struct BondPotential;

struct Foo
{
  Foo(const TriMesh& mesh) :
    mesh_(mesh) {}

  const TriMesh& mesh_;
};

class EnergyManager
{
public:

    // constructors
    EnergyManager(const TriMesh* mesh,
                  const EnergyParams& params);

    // update reference properties
    void update_reference_properties();
    void interpolate_reference_properties();

    // energy and gradient evaluation
    real energy();
    real energy(VertexProperties& props);
    std::vector<Point> gradient();

    // print status information
    void print_info();

    // mesh properties and parameters
    VertexProperties properties;
    EnergyParams params;

    // management of reference properties
    VertexProperties initial_props;
    VertexProperties ref_props;

    // bond potential
    std::unique_ptr<BondPotential> bonds;

    // set mesh
    void set_mesh(const TriMesh* mesh);

private:

    // mesh reference (??)
    const TriMesh* mesh_;
};

void expose_energy(py::module& m);

}
#endif
