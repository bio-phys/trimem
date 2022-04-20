/** \file nlist.cpp
 */

#include "cell_list.h"
#include "verlet_list.h"

#include "pybind11/stl.h"

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

void expose_nlists(py::module& m)
{
    // stump: needed by NeighbourList::point_distances
    py::class_<Point>(
        m,
        "Point",
        R"pbdoc(
        ``OpenMesh::TriMesh::Point``

        Typedef of a vector-like quantity describing vertices/points in 3D.
        )pbdoc"
    );

    // NeighbourList
    py::class_<NeighbourList>(
        m,
        "NeighbourList",
        R"pbdoc(
        Neighbour list interface

        Abstract representation of a neighbour list data structure operating
        on a :class:`TriMesh`. Can be either a ``cell_list`` or a
        ``verlet_list``. See :func:`make_nlist` for its construction.
        )pbdoc"
        )

        .def(
            "distance_matrix",
            &NeighbourList::distance_matrix,
            py::arg("mesh"),
            py::arg("rdist"),
            R"pbdoc(
            Compute sparse distance matrix.

            Args:
                mesh (TriMesh): mesh subject to pair-wise vertex distance
                    compuation
                rdist (float): distance cutoff. This cutoff is additional to the
                    cutoff that was used during list creation! That is, it is
                    meaningful to specify a smaller cutoff than the cutoff used
                    at list creation but a larger cutoff has no effect.

            Returns:
                A sparse distance matrix (as a tuple of lists) containing
                all vertices within a distance ``< rdist``.
            )pbdoc"
        )

        .def(
            "point_distances",
            &NeighbourList::point_distances,
            py::arg("mesh"),
            py::arg("pid"),
            py::arg("rdist"),
            R"pbdoc(
            Compute distances for vertex pid.

            Args:
                mesh (TriMesh): mesh containing vertices to be tested against
                    ``pid``.
                pid (int): vertex in ``mesh`` for which distances to other
                    vertices are to be found.
                rdist (float): distance cutoff. This cutoff is additional to the
                    cutoff that was used during list creation! That is, it is
                    meaningful to specify a smaller cutoff than the cutoff used
                    at list creation but a larger cutoff has no effect.

            Returns:
                A tuple (``distances``, ``ids``) of vectors of distance
                components (in 3D) between ``pid`` and vertices in ``ids``
                with distance ``< rdist``.
            )pbdoc"
        );

    // expose factory
    m.def(
        "make_nlist",
        &make_nlist,
        py::arg("mesh"),
        py::arg("eparams"),
        R"pbdoc(
        Neighbour list factory

        Args:
            mesh (TriMesh): a mesh whose vertices are subject to neighbour search
            eparams (EnergyParams): parametrization of the list structure. Parameters are extracted from the sub-structure :class:`SurfaceRepulsionParams`.

        Returns:
            An interface of type :class:`NeighbourList` representing either
            a cell list or a verlet-list.
        )pbdoc"
    );
}

}
