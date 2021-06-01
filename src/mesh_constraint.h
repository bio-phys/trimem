/** \file mesh_constraint.h
 * \brief Inform on mesh self-intersection using a cell list.
 */
#ifndef MESH_CONSTRAINT_H
#define MESH_CONSTRAINT_H

#include "MeshTypes.hh"

#include "cell_list.h"
#include "neighbour_list.h"

namespace trimem {

//! convenience non-template interface for MeshConstraint
struct IMeshConstraint
{
    virtual bool check_global(const TriMesh& mesh) const = 0;
    virtual bool check_local(const TriMesh& mesh, const int& pid) const = 0;
};

//! Mesh constraint that can use trimem::CellList or trimem::NeighbourLists
// TODO: assert valid template types at compile time
template <class ListType>
struct MeshConstraint : IMeshConstraint
{
    ListType nlist;
    double   d_max;

    MeshConstraint(const TriMesh& mesh, const double& rlist, const double& dmax)
      :
      nlist(mesh, rlist),
      d_max(dmax) {}

    virtual bool check_global(const TriMesh& mesh) const override
    {
        int counts = nlist.distance_counts(mesh, d_max);
        return (counts == 0) ? true : false;
    }

    virtual bool check_local(const TriMesh& mesh, const int& pid) const override
    {
        int counts = nlist.point_distance_counts(mesh, pid, d_max);
        return (counts == 0) ? true : false;
    }
};

// 'instantiate' types to be used
typedef MeshConstraint<trimem::rCellList> MeshConstraintCL;
typedef MeshConstraint<trimem::rNeighbourLists> MeshConstraintNL;

}
#endif
