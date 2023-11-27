/** \file energy_py.cpp
 */
#include "energy_py.h"
#include "numpy_util.h"
#include "mesh_py.h"

namespace trimem {

real energy(
    EnergyManager& estore,
    const py::array_t<typename TriMesh::Point::value_type> points
)
{
    set_points(estore.mesh, points);
    return estore.energy();
}

py::array_t<typename TriMesh::Point::value_type> gradient(
    EnergyManager& estore,
    const py::array_t<typename TriMesh::Point::value_type> points
)
{
    set_points(estore.mesh, points);
    auto grad = estore.gradient();
    return tonumpy(grad[0], grad.size());
}

}
