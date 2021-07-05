/** \file params.cpp
 * \brief Parameters for the helfrich energy.
 */
#include "params.h"

namespace trimem {

void expose_parameters(py::module& m)
{
    py::enum_<BondType>(m, "BondType")
        .value("Edge", BondType::Edge)
        .value("Area", BondType::Area)
        .value("None", BondType::None)
        .export_values();

    py::class_<BondParams>(m, "BondParams")
        .def(py::init())
        .def_readwrite("lc0", &BondParams::lc0)
        .def_readwrite("lc1", &BondParams::lc1)
//        .def_readwrite("lmax", &BondParams::lmax)
//        .def_readwrite("lmin", &BondParams::lmin)
        .def_readwrite("r", &BondParams::r)
        .def_readwrite("a0", &BondParams::a0)
        .def_readwrite("type", &BondParams::type);

    py::class_<EnergyParams>(m, "EnergyParams")
        .def(py::init())
        .def_readwrite("kappa_b", &EnergyParams::kappa_b)
        .def_readwrite("kappa_a", &EnergyParams::kappa_a)
        .def_readwrite("kappa_v", &EnergyParams::kappa_v)
        .def_readwrite("kappa_c", &EnergyParams::kappa_c)
        .def_readwrite("kappa_t", &EnergyParams::kappa_t)

        .def_readwrite("ref_volume", &EnergyParams::ref_volume)
        .def_readwrite("ref_area", &EnergyParams::ref_area)
        .def_readwrite("ref_curvature", &EnergyParams::ref_curvature)

        .def_readwrite("bond_params", &EnergyParams::bond_params);

    py::class_<ContinuationParams>(m, "ContinuationParams")
        .def(py::init())
        .def_readwrite("area_frac", &ContinuationParams::area_frac)
        .def_readwrite("volume_frac", &ContinuationParams::volume_frac)
        .def_readwrite("curvature_frac", &ContinuationParams::curvature_frac)
        .def_readwrite("delta", &ContinuationParams::delta)
        .def_readwrite("lam", &ContinuationParams::lambda);
}

}
