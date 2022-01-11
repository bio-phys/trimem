/** \file flip_utils.h
 * \brief Utility functions for performing edge flips on openmesh.
 */
#ifndef FLIP_UTILS_H
#define FLIP_UTILS_H

#include "defs.h"
#include "mesh_properties.h"

namespace trimem {

class EnergyManager;
class BondPotential;
class SurfaceRepulsion;
class OmpGuard;

//! Compute vertex properties of a mesh patch associated to an edge.
VertexProperties edge_vertex_properties(TriMesh& mesh,
                                       const EdgeHandle& eh,
                                       const BondPotential& bonds,
                                       const SurfaceRepulsion& repulse);

//! Compute neighbourhood patch of edges blocked by a flip of eh
std::unordered_set<int> flip_patch(TriMesh& mesh, const EdgeHandle& eh);

//! "Test" (in the sense of omp_test_lock) locking of edge idx (and it's patch)
bool test_guards(TriMesh& mesh, const int& idx, std::vector<OmpGuard>& locks);

}
#endif
