#include <tuple>
#include <omp.h>

#include "MeshTypes.hh"
#include "OpenMesh/Core/Geometry/VectorT.hh"

#include "pybind11/pybind11.h"

typedef double real;

std::tuple<real, real, real, real> energy(TriMesh& mesh, real kappa)
{
    real curvature = 0;
    real surface   = 0;
    real volume    = 0;
    real energy    = 0;

/*
    for ( TriMesh::HalfedgeIter h_it=mesh.halfedges_begin(); h_it!=mesh.halfedges_end(); ++h_it )
    {
*/
    #pragma omp parallel for reduction(+:curvature,surface,volume,energy)
    for (int i=0; i<mesh.n_halfedges(); i++)
    {
        auto eh = mesh.halfedge_handle(i);
        auto fh = mesh.face_handle(eh);
        if ( fh.is_valid() )
        {
            real sector_area = mesh.calc_sector_area(eh);
            real edge_length = mesh.calc_edge_length(eh);
            real edge_angle  = mesh.calc_dihedral_angle(eh);
            auto face_normal = mesh.calc_face_normal(fh);
            auto face_center = mesh.calc_face_centroid(fh);
            real edge_curv   = 0.5 * edge_angle * edge_length;

            curvature += edge_curv;
            surface   += sector_area;
            volume    += dot(face_normal, face_center) * sector_area / 3;
            energy    += edge_curv * edge_curv;
        }
     }

    // correct multiplicity
    energy    /= 2;
    curvature /= 2;
    surface   /= 3;
    volume    /= 3;
    return std::make_tuple(2 * kappa * energy, surface, volume, curvature);
}

std::tuple<real, real, real, real> energy_v(TriMesh& mesh, real kappa)
{
    real curvature = 0;
    real surface   = 0;
    real volume    = 0;
    real energy    = 0;

    #pragma omp parallel for reduction(+:curvature,surface,volume,energy)
    for (int i=0; i<mesh.n_vertices(); i++)
    {
        real c=0.0, s=0.0, v = 0.0;
        auto ve = mesh.vertex_handle(i);
        for(TriMesh::VertexOHalfedgeIter h_it = mesh.voh_iter(ve); h_it; ++h_it)
        {
            auto fh = mesh.face_handle(*h_it);

            if ( fh.is_valid() )
            {
                real sector_area = mesh.calc_sector_area(*h_it);
                real edge_length = mesh.calc_edge_length(*h_it);
                real edge_angle  = mesh.calc_dihedral_angle(*h_it);
                auto face_normal = mesh.calc_face_normal(fh);
                auto face_center = mesh.calc_face_centroid(fh);
                real edge_curv   = 0.25 * edge_angle * edge_length;

                c += edge_curv;
                s += sector_area / 3;
                v += dot(face_normal, face_center) * sector_area / 3;
            }
        }

        surface   += s;
        volume    += v;
        curvature += c;
        energy    += 2 * kappa * c * c / s;
     }

    // correct multiplicity
    volume    /= 3;
    return std::make_tuple(energy, surface, volume, curvature);
}

int check_edge_lengths(TriMesh& mesh, real min_tether, real max_tether)
{
    int invalid_edges = 0;

    #pragma omp parallel for reduction(+:invalid_edges)
    for (int i=0; i<mesh.n_edges(); i++)
    {
        auto eh = mesh.edge_handle(i);
        auto el = mesh.calc_edge_length(eh);
        if ((el<min_tether) or (el>max_tether))
            invalid_edges += 1;
    }

    return invalid_edges;
}

int flip_edges(TriMesh& mesh)
{
    int flips = 0;
    for (int i=0; i<mesh.n_edges(); i++)
    {
        auto eh = mesh.edge_handle(i);
        real el = mesh.calc_edge_length(eh);
        if (mesh.is_flip_ok(eh))
        {
            mesh.flip(eh);
            real el_new = mesh.calc_edge_length(eh);
            if (el_new > el)
            {
                mesh.flip(eh);
            }
            else
                flips += 1;
        }
    }

    return flips;
}

PYBIND11_MODULE(test, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("calc_energy", &energy, "Edge-based evaluation of the helfrich energy");
    m.def("calc_energy_v", &energy_v, "Vertex-based evaluation of the helfrich energy");
    m.def("check_edges", &check_edge_lengths, "Check for invalid edges");
    m.def("flip_edges", &flip_edges, "Flip edges if convenient");
}
