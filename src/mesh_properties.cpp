/** \file mesh_properties.cpp
 * \brief Geometric properties and gradients on a OpenMesh::TriMesh.
 */
#include "mesh_properties.h"

#include "mesh_util.h"
#include "mesh_tether.h"
#include "mesh_repulsion.h"

namespace trimem {

VertexProperties vertex_properties(const TriMesh& mesh,
                                   const BondPotential& bonds,
                                   const SurfaceRepulsion& constraint,
                                   const VertexHandle& ve)
{
    VertexProperties p{ 0, 0, 0, 0, 0, 0 };

    for (auto he : mesh.voh_range(ve))
    {
        if ( not he.is_boundary() )
        {
            // edge curvature
            real el      = edge_length(mesh, he);
            real da      = dihedral_angle(mesh, he);
            p.curvature += 0.5 * el * da;

            // face area/volume
            p.area   += face_area(mesh, he);
            p.volume += face_volume(mesh, he);

            // bonds
            p.tethering += bonds.vertex_property(mesh, he);
        }
    }

    // correct multiplicity
    p.area      /= 3;
    p.volume    /= 3;
    p.curvature /= 2;
    p.tethering /= bonds.valence();

    p.bending    = 2 * p.curvature * p.curvature / p.area;

    // mesh repulsion
    p.repulsion = constraint.vertex_property(mesh, ve.idx());

    return p;
}

void vertex_properties_grad(const TriMesh& mesh,
                            const BondPotential& bonds,
                            const SurfaceRepulsion& repulse,
                            const VertexHandle& ve,
                            const std::vector<VertexProperties>& props,
                            std::vector<VertexPropertiesGradient>& d_props)
{
    // This routine, instead of evaluating the gradient of each vertex-averaged
    // property wrt the coordinates of all the vertices involved (which would
    // require atomic operations or some gradient buffers; the former having
    // shown to produce a significant slow down, the latter being inefficient for
    // large number of threads), gathers gradients from all vertex-averaged
    // properties ve is involved in, i.e., summing only into the gradient at index
    // ve.idx() (being not used by other threads!).

    // pre-compute vertex-curvature to vertex-area ratio (needed for bending)
    // for the patch of vertex ve (at pos 0 of this array) and later also at
    // pos 1 and 2 for other vertex-patches.
    std::array<real, 3> c_to_a;
    auto idx = ve.idx();
    c_to_a[0] = props[idx].curvature / props[idx].area;

    for (auto he : mesh.voh_range(ve))
    {
        if ( not he.is_boundary() )
        {
            auto n_he = mesh.next_halfedge_handle(he);
            auto jdx  = mesh.to_vertex_handle(he).idx();
            auto kdx  = mesh.to_vertex_handle(n_he).idx();

            // vertex-curvature ratio of other patches
            c_to_a[1] = props[jdx].curvature / props[jdx].area;
            c_to_a[2] = props[kdx].curvature / props[kdx].area;

            // edge curvature of outgoing he
            // the gradient of the edge-length as well as the dihedral-angle
            // is symmetric wrt to edge-swap, i.e. when seen from vertex jdx
            real edge_length = trimem::edge_length(mesh, he);
            real edge_angle  = trimem::dihedral_angle(mesh, he);
            auto d_length    = trimem::edge_length_grad<1>(mesh, he);
            auto d_angle     = trimem::dihedral_angle_grad<1>(mesh, he);
            for (int i=0; i<2; i++)
            {
                auto val = 0.25 * (edge_angle * d_length[0] +
                                   edge_length * d_angle[0]);
                d_props[idx].curvature += val;
                d_props[idx].bending   += 4.0 * c_to_a[i] * val;
            }

            // face area of outgoing he
            // contribution from self as well as from outgoing he of jdx,
            // the latter being equivalent just needs different c_to_a
            auto d_face_area = trimem::face_area_grad<1>(mesh, he);
            for (int i=0; i<3; i++)
            {
                auto val = d_face_area[0] / 3;
                d_props[idx].area    += val;
                d_props[idx].bending -= 2.0  * c_to_a[i] * c_to_a[i] * val;
            }

            // face volume of outging he
            // same logic as for the area
            auto d_face_volume = trimem::face_volume_grad<1>(mesh, he);
            for (size_t i=0; i<3; i++)
            {
                auto val = d_face_volume[0] / 3;
                d_props[idx].volume += val;
            }

            // edge curvature of next halfedge
            edge_length = trimem::edge_length(mesh, n_he);
            edge_angle  = trimem::dihedral_angle(mesh, n_he);
            d_angle     = trimem::dihedral_angle_grad<4>(mesh, n_he);
            for (int i=1; i<3; i++)
            {
                auto val = 0.25 * edge_length * d_angle[0];
                d_props[idx].curvature += val;
                d_props[idx].bending   += 4.0 * c_to_a[i] * val;
            }

            // tether bonds
            auto d_bond = bonds.vertex_property_grad(mesh, he)[0];
            d_props[idx].tethering += d_bond;

        } // not boundary
    } // outgoing halfedges

    // repulsion penalty
    std::vector<Point> d_repulse;
    std::vector<int> jdx;
    std::tie(d_repulse, jdx) = repulse.vertex_property_grad(mesh, idx);
    for (size_t i=0; i<d_repulse.size(); i++)
    {
        auto val = d_repulse[i];
        d_props[idx].repulsion += 2 * d_repulse[i];
    }

}
}
