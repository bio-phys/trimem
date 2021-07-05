/** \file numpy_util.h
 * \brief c++ to numpy utils.
 */
#ifndef NUMPY_UTIL_H
#define NUMPY_UTIL_H

#include "pybind11/numpy.h"

namespace trimem {

// vector of some array-type to numpy
template<class Row>
py::array_t<typename Row::value_type> tonumpy(Row& _vec, size_t _n = 1) {
	typedef typename Row::value_type dtype;
	std::vector<size_t> shape;
	std::vector<size_t> strides;
	if (_n == 1) {
		shape = {_vec.size()};
		strides = {sizeof(dtype)};
	}
	else {
		shape = {_n, _vec.size()};
		strides = {_vec.size() * sizeof(dtype), sizeof(dtype)};
	}
	return py::array_t<dtype>(shape, strides, _vec.data());
}

}
#endif
