/** \file params.h
 * \brief Parameters for the helfrich energy.
 */
#ifndef PARAMS_H
#define PARAMS_H

#include <optional>

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

    std::string               pathspec;
    std::vector<real>         pathdata;
    std::optional<std::string> label;

    real lambda;
    real delta;
    real state;

    void eval_state()
    {
        // start-stop
        if ( pathdata.size() == 2 )
            state = (1 - lambda) * pathdata[0] + lambda * pathdata[1];
        // constant
        else if ( pathdata.size() == 1 )
            state = pathdata[0];
        // discrete path
        else if ( pathdata.size() > 2 )
        {
            auto N  = pathdata.size();
            auto si = 1.0 / ( N - 1 );
            real r  = lambda / si;
            int  il = r;
            if (il == N) state = pathdata[N-1];
            int  iu = il + 1;
            real g  = r - il;

            state = pathdata[il] + g * ( pathdata[iu] - pathdata[il]);
        }
    }

public:

    ContinuationTuple(
        const real& start,
        const real& stop,
        const real& delta,
        const real& lambda
    ) :
        delta(delta),
        lambda(lambda)
    {
        pathdata.resize(2);
        pathdata[0] = start;
        pathdata[1] = stop;
        eval_state();
    }

    ContinuationTuple(
        const std::vector<real> data,
        const std::string& spec,
        const real& delta,
        const real& lambda,
        std::optional<std::string> label
    ) :
        pathdata(data),
        pathspec(spec),
        delta(delta),
        lambda(lambda),
        label(label)
    {
        eval_state();
    }

    ContinuationTuple(const real& start)
    {
        pathdata.resize(1);
        pathdata[0] = start;
        eval_state();
    }

    void update()
    {
        if (lambda < 1) lambda += delta;
        if (lambda > 1) lambda = 1;
        eval_state();
    }

    operator const real&() const {return state;}
    operator autodiff::real() const {return state;}

    std::string to_string() const {
        if ( pathdata.size() == 2 )
            return std::to_string(pathdata[0]) + " " + \
                   std::to_string(pathdata[1]) + " " + \
                   std::to_string(delta) + " " + \
                   std::to_string(lambda);
        else if (pathdata.size() == 1)
            return std::to_string(pathdata[0]);
        else if (pathdata.size() > 2)
        {
            std::string out = "$<" + pathspec + ">$ " + \
                   std::to_string(pathdata.size()) + " " + \
                   std::to_string(delta) + " " + \
                   std::to_string(lambda);
            if (label.has_value()) out += " " + label.value();
            return out;
        }
        else
            return "unknown";
    }
};

struct SurfaceRepulsionParams
{
  real        lc1             = 0;
  int         r               = 2;
  std::string n_search        = "cell-list";
  real        rlist           = 0.1;
  int         exclusion_level = 2;
  int         refresh         = 1;
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
  ContinuationTuple area_fraction = 1;
  //! target volume as fraction of initial volume
  ContinuationTuple volume_fraction = 1;
  //! target curvature as fraction of initial curvature
  ContinuationTuple curvature_fraction = 1;

  //! parameters for the tether penalty
  BondParams bond_params;
  //! parameters for the repulsion penalty
  SurfaceRepulsionParams repulse_params;
  //! parametees for the external potential
  ExternalPotentialParams external_params;
};

}
#endif
