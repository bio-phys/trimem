/** \file energy_py.h
 * \brief python specific bindings for energy.
 */
#ifndef ENERGY_PY_H
#define ENERGY_PY_H

#include "energy.h"

#include "pybind11/numpy.h"

namespace py = pybind11;

namespace trimem {

real energy(
    EnergyManager& estore,
    const py::array_t<typename TriMesh::Point::value_type> points
);

py::array_t<typename TriMesh::Point::value_type> gradient(
    EnergyManager& estore,
    const py::array_t<typename TriMesh::Point::value_type> points
);

}
#endif
