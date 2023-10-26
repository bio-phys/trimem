/** \file params.h
 * \brief Parameters for the helfrich energy.
 */
#ifndef PARAMS_H
#define PARAMS_H

#include "defs.h"

#include <autodiff/forward/real.hpp>

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
  real lc0  = 1;
  real lc1  = 0;
  real a0   = 1;
  int  r    = 2;
  BondType type = BondType::Edge;
};

class ContinuationTuple
{
private:

    real start;
    real stop;
    real delta;
    real lambda;

    real state;

    void eval_state()
    {
        state = (1 - lambda) * start + lambda * stop;
    }

public:

    ContinuationTuple(
        const real& start,
        const real& stop,
        const real& delta,
        const real& lambda
    ) :
        start(start),
        stop(stop),
        delta(delta),
        lambda(lambda)
    {
        eval_state();
    }

    ContinuationTuple(const real& start) :
        start(start),
        stop(start),
        delta(0),
        lambda(0),
        state(start) {}

    void update()
    {
        if (lambda < 1) lambda += delta;
        if (lambda > 1) lambda = 1;
        eval_state();
    }

    operator const real&() const {return state;}
    operator autodiff::real() const {return state;}

    std::string to_string() const {
        return std::to_string(start) + " " + std::to_string(stop) + " " + \
               std::to_string(delta) + " " + std::to_string(lambda);
    }
};

struct SurfaceRepulsionParams
{
  real        lc1             = 0;
  int         r               = 2;
  std::string n_search        = "cell-list";
  real        rlist           = 0.1;
  int         exclusion_level = 2;
};

struct ExternalPotentialParams
{
  std::string type              = "none";
  real epsilon                  = 1.0;
  real sigma                    = 1.0;
  ContinuationTuple radius      = 1.0;
};

struct EnergyParams
{
  //! weight bending energy
  real kappa_b = 0;
  //! weight area penalty
  real kappa_a = 0;
  //! weight volume penalty
  real kappa_v = 0;
  //! weight area-diff penalty
  real kappa_c = 0;
  //! weight tether penalty
  real kappa_t = 0;
  //! weight repulsion penalty
  real kappa_r = 0;
  //! weight external potential
  real kappa_e = 0;

  //! target area as fraction of initial area
  ContinuationTuple area_frac = 1;
  //! target volume as fraction of initial volume
  ContinuationTuple volume_frac = 1;
  //! target curvature as fraction of initial curvature
  ContinuationTuple curvature_frac = 1;

  //! parameters for the tether penalty
  BondParams bond_params;
  //! parameters for the repulsion penalty
  SurfaceRepulsionParams repulse_params;
  //! parametees for the external potential
  ExternalPotentialParams external_params;
};

}
#endif
