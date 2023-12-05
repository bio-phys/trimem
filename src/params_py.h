/** \file params_py.h
 * \brief Utilities for the python bindings of params.
 */
#ifndef PARAMS_PY_H
#define PARAMS_PY_H

#include "params.h"

#include "pybind11/pybind11.h"

namespace py = pybind11;

namespace trimem{

ContinuationTuple make_continuation_from_list(const py::list& args);

BondType make_bondtype(const std::string& type);

}
#endif
