/** \file nlist.h
 * \brief Neighbour list interface for the algorithms to be used with openmesh.
 */
#ifndef NLIST_H
#define NLIST_H

#include <vector>

#include "defs.h"
#include "params.h"

#include "pybind11/pybind11.h"

namespace trimem {

//! Neighbour list interface
struct NeighbourList
{
  virtual std::tuple<std::vector<real>, std::vector<int>, std::vector<int> >
  distance_matrix(const TriMesh& mesh, const real& dmax) const = 0;

  virtual std::tuple<std::vector<Point>, std::vector<int> >
  point_distances(const TriMesh& mesh, const int& pid, const real& dmax) const = 0;
};

//! Neighbour list templated base class
template<int exclusion = 0>
struct NeighbourListT : NeighbourList
{
  virtual std::tuple<std::vector<real>, std::vector<int>, std::vector<int> >
  distance_matrix(const TriMesh& mesh, const real& dmax) const = 0;

  virtual std::tuple<std::vector<Point>, std::vector<int> >
  point_distances(const TriMesh& mesh, const int& pid, const real& dmax) const = 0;
};

std::unique_ptr<NeighbourList> make_nlist(const TriMesh& mesh,
                                          const EnergyParams& params);

void expose_nlists(py::module& m);

}
#endif
