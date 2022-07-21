/** \file util.cpp
 * \brief Some generic trime utilities.
 */
#include "util.h"
#include "mesh_util.h"

namespace trimem {

real area(const TriMesh& mesh)
{
    real area = 0.0;

    #pragma omp parallel for reduction(+:area)
    for (size_t i=0; i<mesh.n_faces(); i++)
    {
        auto he = mesh.halfedge_handle(mesh.face_handle(i));
        area += face_area(mesh, he);
    }

    return area;
}

real edges_length(const TriMesh& mesh)
{
    real length = 0.0;

    #pragma omp parallel for reduction(+:length)
    for (size_t i=0; i<mesh.n_edges(); i++)
    {
        auto he = mesh.halfedge_handle(mesh.edge_handle(i),0);
        length += edge_length(mesh, he);
    }

    return length;
}

std::tuple<real, real> mean_tri_props(const TriMesh& mesh)
{
    return std::make_tuple<real, real>(area(mesh)/mesh.n_faces(),
                                       edges_length(mesh)/mesh.n_edges());
}

}
