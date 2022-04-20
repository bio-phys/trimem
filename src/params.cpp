/** \file params.cpp
 * \brief Parameters for the helfrich energy.
 */
#include "params.h"

namespace trimem {

void expose_parameters(py::module& m)
{
    // these classes are POD, so no need for __init__ signatures from python
    py::options options;
    options.disable_function_signatures();

    py::enum_<BondType>(
        m,
        "BondType",
        "Types for the tether potential."
        )
        .value("Edge", BondType::Edge, "Smoothed well/box potential on edges.")
        .value("Area", BondType::Area, "Harmonic potential on face area.")
        .value("None", BondType::None, "None")
        .export_values();

    py::class_<BondParams>(
        m,
        "BondParams",
        "Tether regularization parameters."
        )
        .def(py::init())
        .def_readwrite(
            "lc0",
            &BondParams::lc0,
            R"pbdoc(
            Onset distance for attracting force (for BondType::Edge).

            :type: float
            )pbdoc"
        )
        .def_readwrite("lc1",
            &BondParams::lc1,
            R"pbdoc(
            Onset distance for repelling force (for BondType::Edge).

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "a0",
            &BondParams::a0,
            R"pbdoc(
            Reference face area (for BondType::Area).

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "r",
            &BondParams::r,
            R"pbdoc(
            Steepness of regularization potential (must be >=1).

            :type: int.
            )pbdoc"
        )
        .def_readwrite(
            "type",
            &BondParams::type,
            R"pbdoc(
            Type of potential (edge-based, area-based).

            :type: BondType
            )pbdoc"
         );

    py::class_<SurfaceRepulsionParams>(
        m,
        "SurfaceRepulsionParams",
        "Parameters for the surface repulsion penalty."
        )
        .def(py::init())
        .def_readwrite(
            "lc1",
            &SurfaceRepulsionParams::lc1,
            R"pbdoc(
            Onset distance for repelling force.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "r",
            &SurfaceRepulsionParams::r,
            R"pbdoc(
            Steepness of repelling potential (must be >=1).

            :type: int.
            )pbdoc"
        )
        .def_readwrite(
            "n_search",
            &SurfaceRepulsionParams::n_search,
            R"pbdoc(
            Type of neighbour list structures.

            :type: str

            Can be ``cell_list`` or ``verlet_list``.
            )pbdoc"
        )
        .def_readwrite(
            "rlist",
            &SurfaceRepulsionParams::rlist,
            "Neighbour search distance cutoff."
        )
        .def_readwrite(
            "exclusion_level",
            &SurfaceRepulsionParams::exclusion_level,
            R"pbdoc(
            Connected neighbourhood exclusion for neighbour lists.

            :type: int

            Levels of exclusion are inclusive, i.e. 0<1<2. Can be one of:
                * 0: exclude self
                * 1: exclude directly connected neighbourhood (1 edge)
                * 2: exclude indirectly connected neighbourhood (2 edges)

            )pbdoc"
        );

    py::class_<ContinuationParams>(
        m,
        "ContinuationParams",
        "Parameters used for smooth continuation."
        )
        .def(py::init())
        .def_readwrite(
            "delta",
            &ContinuationParams::delta,
            "Interpolation blending `time` step."
        )
        .def_readwrite(
            "lam",
            &ContinuationParams::lambda,
            "Interpolation state."
        );

    py::class_<EnergyParams>(
        m,
        "EnergyParams",
        R"pbdoc(
        Parametrization of the Helfrich functional.

        Modularized POD structure containing parameters for the Helfrich
        functional, the `area`, `volume` and `area-difference` penalties,
        the repulsion penalty and the tether regularization.
        )pbdoc"
        )

        .def(py::init())
        .def_readwrite(
            "kappa_b",
            &EnergyParams::kappa_b,
            R"pbdoc(
            Weight of the Helfrich functional.

            :type: float
            )pbdoc"
         )
        .def_readwrite(
            "kappa_a",
            &EnergyParams::kappa_a,
            R"pbdoc(
            Weight of the surface area penalty.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "kappa_v",
            &EnergyParams::kappa_v,
            R"pbdoc(
            Weight of the volume penalty.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "kappa_c",
            &EnergyParams::kappa_c,
            R"pbdoc(
            Weight of the area-difference penalty.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "kappa_t",
            &EnergyParams::kappa_t,
            R"pbdoc(
            Weight of the tether regularization.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "kappa_r",
            &EnergyParams::kappa_r,
            R"pbdoc(
            Weight of the surface repulsion penalty.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "area_frac",
            &EnergyParams::area_frac,
            R"pbdoc(
            Target surface area fraction wrt. the initial geometry.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "volume_frac",
            &EnergyParams::volume_frac,
            R"pbdoc(
            Target volume fraction wrt. the initial geometry.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "curvature_frac",
            &EnergyParams::curvature_frac,
            R"pbdoc(
            Target curvature fraction wrt. the initial geometry.

            :type: float
            )pbdoc"
        )
        .def_readwrite(
            "bond_params",
            &EnergyParams::bond_params,
            R"pbdoc(
            Parameters for the tether regularization.

            :type: BondParams
            )pbdoc"
        )
        .def_readwrite(
            "repulse_params",
            &EnergyParams::repulse_params,
            R"pbdoc(
            Parameters for the surface repulsion.

            :type: SurfaceRepulsionParams
            )pbdoc"
        )
        .def_readwrite(
            "continuation_params",
            &EnergyParams::continuation_params,
            R"pbdoc(
            Parameters for the parameter continuation.

            :type: ContinuationParams
            )pbdoc"
        );

}

}
