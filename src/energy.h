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

    EnergyManager(const TriMesh* mesh,
                  const EnergyParams& params,
                  const ContinuationParams& cparams);

    // update reference properties
    void update_reference_properties();

    // energy and gradient evaluation
    real energy();
    std::vector<Point> gradient();

    // print status information
    void print_info();

    // mesh properties and parameters
    VertexProperties properties;
    EnergyParams params;

private:

    // mesh reference (??)
    const TriMesh* mesh_;

    // bond potential
    std::unique_ptr<BondPotential> bonds_;

    // reference properties' management
    ContinuationParams cparams_;
    VertexProperties initial_props_;
    VertexProperties target_props_;

    bool init_ = false;
};

void expose_energy(py::module& m);

}
#endif
