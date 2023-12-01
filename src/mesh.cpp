/** \file mesh.cpp
 */
#include "mesh.h"

namespace trimem {

TriMesh read_mesh(const std::string fname)
{
    TriMesh mesh;
    if (!OpenMesh::IO::read_mesh(mesh, fname))
    {
        std::cerr << "read error on file " << fname << "\n";
        exit(1);
    }
    return mesh;
}

void write_mesh(const TriMesh& mesh, const std::string fname)
{
    OpenMesh::IO::Options opt;
    opt += OpenMesh::IO::Options::Custom;

    if (!OpenMesh::IO::write_mesh(mesh, fname, opt))
    {
        std::cerr << "write error on file " << fname << "\n";
        exit(1);
    }
}

}
