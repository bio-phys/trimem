#include <omp.h>

#include "MeshTypes.hh"
#include "OpenMesh/Core/Geometry/VectorT.hh"

#include "pybind11/pybind11.h"

typedef double real;

double energy(TriMesh& mesh, real kappa)
{
    real curvature = 0;
    real surface   = 0;
    real volume    = 0;
    real energy    = 0;

/*
    for ( TriMesh::HalfedgeIter h_it=mesh.halfedges_begin(); h_it!=mesh.halfedges_end(); ++h_it )
    {
*/
    #pragma omp parallel for reduction(+:curvature,surface,volume, energy)
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
    return 2 * kappa * energy; //surface, volume, curvature

}

PYBIND11_MODULE(test, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("print_mesh", &energy, "A function which adds two numbers");
}
