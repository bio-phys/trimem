/** \file flips.h
 * \brief Performing edge flips on openmesh.
 */
#ifndef FLIPS_H
#define FLIPS_H

#include "defs.h"

#include "mesh.h"

namespace trimem {

struct EnergyManager;

struct EnergyManagerNSR;

int flip_serial(TriMesh& mesh, EnergyManager& estore, const real& flip_ratio);

int flip_parallel_batches(TriMesh& mesh, EnergyManager& estore, const real& flip_ratio);

int flip_serial_nsr(TriMesh& mesh, EnergyManagerNSR& estore, const real& flip_ratio);

std::vector<std::array<int,4>> flip_parallel_batches_nsr(TriMesh& mesh, EnergyManagerNSR& estore, const real& flip_ratio);
}
#endif
