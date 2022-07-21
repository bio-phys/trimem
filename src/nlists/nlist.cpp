/** \file nlist.cpp
 */

#include "cell_list.h"
#include "verlet_list.h"

namespace trimem {

std::unique_ptr<NeighbourList> make_nlist(const TriMesh& mesh,
                                          const EnergyParams& params)
{
    auto& list_type = params.repulse_params.n_search;
    auto& excl      = params.repulse_params.exclusion_level;
    auto& rlist     = params.repulse_params.rlist;

    if (list_type == "cell-list")
    {
        if (excl == 0)
        {
            return std::make_unique<CellList<0>>(mesh, rlist);
        }
        else if (excl == 1)
        {
            return std::make_unique<CellList<1>>(mesh, rlist);
        }
        else if (excl == 2)
        {
            return std::make_unique<CellList<2>>(mesh, rlist);
        }
        else
            throw std::runtime_error("Unsupported exclusion level");
    }
    else if (list_type == "verlet-list")
    {
        if (excl == 0)
        {
            return std::make_unique<VerletList<0>>(mesh, rlist);
        }
        else if (excl == 1)
        {
            return std::make_unique<VerletList<1>>(mesh, rlist);
        }
        else if (excl == 2)
        {
            return std::make_unique<VerletList<2>>(mesh, rlist);
        }
        else
            throw std::runtime_error("Unsupported exclusion level");
    }
    else
        throw std::runtime_error("Unknown neighbour search algorithm.");
}

}
