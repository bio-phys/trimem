/** \file mesh_properties.h
 * \brief Evaluate vertex-based properties.
 */
#ifndef MESH_PROPERTIES_H
#define MESH_PROPERTIES_H

#include "defs.h"

#include "pybind11/pybind11.h"

namespace trimem {

struct BondPotential;

template<class T>
struct TVertexProperties
{
    T area;
    T volume;
    T curvature;
    T bending;
    T tethering;

    TVertexProperties<T>& operator+=(const TVertexProperties<T>& lhs)
    {
        area      += lhs.area;
        volume    += lhs.volume;
        curvature += lhs.curvature;
        bending   += lhs.bending;
        tethering += lhs.tethering;
        return *this;
    }

    TVertexProperties<T>& operator-=(const TVertexProperties<T>& lhs)
    {
        area      -= lhs.area;
        volume    -= lhs.volume;
        curvature -= lhs.curvature;
        bending   -= lhs.bending;
        tethering -= lhs.tethering;
        return *this;
    }
};

typedef TVertexProperties<real>  VertexProperties;
typedef TVertexProperties<Point> VertexPropertiesGradient;

VertexProperties vertex_properties(const TriMesh& mesh,
                                   const BondPotential& bonds,
                                   const VertexHandle& ve);

void vertex_properties_grad(const TriMesh& mesh,
                            const BondPotential& bonds,
                            const VertexHandle& ve,
                            std::vector<VertexPropertiesGradient>& d_props);

void expose_properties(py::module& m);

}
#endif
