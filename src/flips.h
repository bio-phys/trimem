/** \file flips.h
 * \brief Performing edge flips on openmesh.
 */
#ifndef FLIPS_H
#define FLIPS_H

#include "defs.h"

#include "mesh.h"

namespace trimem {

struct EnergyManager;

int flip_serial(EnergyManager& estore, const real& flip_ratio);

int flip_parallel_batches(EnergyManager& estore, const real& flip_ratio);
}
#endif
