/** \file mesh_properties.h
 * \brief Evaluate vertex-based properties.
 */
#ifndef MESH_PROPERTIES_H
#define MESH_PROPERTIES_H

#include "defs.h"

namespace trimem {

struct BondPotential;
struct SurfaceRepulsion;

template<class T>
struct TVertexProperties
{
    T area;
    T volume;
    T curvature;
    T bending;
    T tethering;
    T repulsion;



    TVertexProperties<T>& operator+=(const TVertexProperties<T>& lhs)
    {
        area      += lhs.area;
        volume    += lhs.volume;
        curvature += lhs.curvature;
        bending   += lhs.bending;
        tethering += lhs.tethering;
        repulsion += lhs.repulsion;
        return *this;
    }

    TVertexProperties<T>& operator-=(const TVertexProperties<T>& lhs)
    {
        area      -= lhs.area;
        volume    -= lhs.volume;
        curvature -= lhs.curvature;
        bending   -= lhs.bending;
        tethering -= lhs.tethering;
        repulsion -= lhs.repulsion;
        return *this;
    }
};

typedef TVertexProperties<real>  VertexProperties;
typedef TVertexProperties<Point> VertexPropertiesGradient;

/*! Evaluate VertexProperties
 *
 *  Evaluates the vertex averaged properties defined on a patch
 *  of faces that are connected to the vertex pointed to by handle ve.
 *
 *  \param[in] mesh       TriMesh instance
 *  \param[in] bonds      description of the bond potential
 *  \param[in] constraint description of the surface repulsion constraint
 *  \param[in] ve         VertexHandle defining the patch of faces
 *  \return an instance of VertexProperties
 */
VertexProperties vertex_properties(const TriMesh& mesh,
                                   const BondPotential& bonds,
                                   const SurfaceRepulsion& constraint,
                                   const VertexHandle& ve);

/*! Evaluate gradient of VertexProperties wrt vertex positions
 *
 *  Evaluates the gradient of the function vertex_properties wrt to
 *  to the coordinates of the involved vertices.
 *
 *  \param[in]  mesh       TriMesh instance
 *  \param[in]  bonds      description of the bond potential
 *  \param[in]  constraint description of the surface repulsion constraint
 *  \param[in]  ve         VertexHandle defining the patch of faces
 *  \param[in]  props      Pre-evaluated vector of vertex-averages properties
 *  \param[out] d_props    Vector of the gradients of the vertex-averaged
 *                         properties
 */
void vertex_properties_grad(const TriMesh& mesh,
                            const BondPotential& bonds,
                            const SurfaceRepulsion& constraint,
                            const VertexHandle& ve,
                            const std::vector<VertexProperties>& props,
                            std::vector<VertexPropertiesGradient>& d_props);



}




#endif
