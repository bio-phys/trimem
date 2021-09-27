/** \file flips.h
 * \brief Performing edge flips on openmesh.
 */
#ifndef FLIPS_H
#define FLIPS_H

#include "defs.h"
#include "pybind11/pybind11.h"

namespace trimem {

void expose_flips(py::module& m);

}
#endif
