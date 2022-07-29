/* Mini c++ wrapper of the core functionality provided by the c++-module
 *
 * Useful for debugging and profiling without all the python interpreter
 * obfuscation.
 */
#include <iostream>
#include <stdlib.h>

#include "defs.h"

#include "mesh.h"
#include "params.h"
#include "energy.h"
#include "util.h"
#include "mesh_tether.h"
#include "nlists/nlist.h"
#include "mesh_repulsion.h"
#include "flips.h"

int main(int argc, char* argv[])
{
    if (argc < 2)
    {
        std::cerr << "Give input file as first positional arg\n";
        exit(1);
    }
    if (argc != 3)
    {
        std::cerr << "Give rlist as second positional arg\n";
        exit(1);
    }

    // read mesh
    std::string fname = argv[1];
    trimem::TriMesh mesh = trimem::read_mesh(fname);

    double rlist = atof(argv[2]);

    // pre-evaluate characteristic bond length
    auto l = std::get<1>(trimem::mean_tri_props(mesh));

    trimem::BondParams bonds;
    bonds.lc0 = 1.25 * l;
    bonds.lc1 = 0.75 * l;
    trimem::SurfaceRepulsionParams repulsion;
    repulsion.lc1      = 0.75 * l;
    repulsion.rlist    = rlist;
    repulsion.n_search = "verlet-list";

    trimem::EnergyParams params;
    params.kappa_b = 300;
    params.kappa_a = 1.0e6;
    params.kappa_v = 1.0e6;
    params.kappa_c = 0;
    params.kappa_t = 1.0e5;
    params.kappa_r = 1.0e3;

    params.bond_params = bonds;
    params.repulse_params = repulsion;

    trimem::EnergyManager estore(mesh, params);

    auto e = estore.energy(mesh);
    auto g = estore.gradient(mesh);
    auto n = flip_serial(mesh, estore, 0.1);

    auto N = g.size();
    std::cout << "Energy: " << e << "\n";
    std::cout << "Gradient: \n[[" << g[0] << "]\n ...\n[" << g[N-1] << "]]\n";
    std::cout << "Flipped " << n << " triangles\n";

    return 0;
}
