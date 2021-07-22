/** \file params.h
 * \brief Parameters for the helfrich energy.
 */
#ifndef PARAMS_H
#define PARAMS_H

#include "defs.h"

#include "pybind11/pybind11.h"

namespace trimem {

enum class BondType : int
{
  Edge,
  Area,
  None,
  Default = Edge
};

struct BondParams
{
  real lc0  = 1.0;
  real lc1  = 0.0;
//  real lmax = 1.0;
//  real lmin = 0.0;
  real a0   = 1.0;
  int  r    = 2;
  BondType type = BondType::Edge;
};

struct EnergyParams
{
  real kappa_b = 1.0;
  real kappa_a = 1.0;
  real kappa_v = 1.0;
  real kappa_c = 1.0;
  real kappa_t = 1.0;

  real area_frac;
  real volume_frac;
  real curvature_frac;

  BondParams bond_params;
};

struct ContinuationParams
{
    real delta = 0.0;
    real lambda = 1.0;
};


void expose_parameters(py::module& m);

}
#endif
