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

}
