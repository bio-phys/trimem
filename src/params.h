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

struct ContinuationParams
{
    real delta  = 0.0;
    real lambda = 1.0;
};

struct SurfaceRepulsionParams
{
  real        lc1             = 0.0;
  int         r               = 2;
  std::string n_search        = "cell-list"; //verlet-list
  real        rlist           = 0.1;
  int         exclusion_level = 2;
};

struct EnergyParams
{
  //! weight bending energy
  real kappa_b = 1.0;
  //! weight area penalty
  real kappa_a = 1.0;
  //! weight volume penalty
  real kappa_v = 1.0;
  //! weight area-diff penalty
  real kappa_c = 1.0;
  //! weight tether penalty
  real kappa_t = 1.0;
  //! weight repulsion penalty
  real kappa_r = 1.0;

  //! target area as fraction of initial area
  real area_frac;
  //! target volume as fraction of initial volume
  real volume_frac;
  //! target curvature as fraction of initial curvature
  real curvature_frac;

  //! parameters for the tether penalty
  BondParams bond_params;
  //! parameters for the interpolation of reference properties
  ContinuationParams continuation_params;
  //! parameters for the repulsion penalty
  SurfaceRepulsionParams repulse_params;
};

void expose_parameters(py::module& m);

}
#endif
