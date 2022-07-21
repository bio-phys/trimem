/** \file util.h
 * \brief Some generic trime utilities.
 */
#ifndef UTIL_H
#define UTIL_H

#include "defs.h"

#include "mesh.h"

namespace trimem {

struct EnergyManager;

real area(const TriMesh& mesh);

real edges_length(const TriMesh& mesh);

std::tuple<real, real> mean_tri_props(const TriMesh& mesh);

}
#endif
