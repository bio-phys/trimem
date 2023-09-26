/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.

   This class implements a pair style for non-reciprocal interactions.
   The shape of the potential is inspired by Soto & Golestanian, PRL 068301 (2014)
   and Soto & Golestanian, PRE 052304 (2015). However, here the user has the option
   of defining the exponent of the interaction, making it longer or shorter range.

------------------------------------------------------------------------- */


#ifdef PAIR_CLASS

  PairStyle(nonreciprocal,PairNonReciprocal);

#else

  #ifndef LMP_PAIR_NONRECIPROCAL_H

    #define LMP_PAIR_NONRECIPROCAL_H

    #include "pair.h"

    namespace LAMMPS_NS {

      class PairNonReciprocal : public Pair {

       public:

        PairNonReciprocal(class LAMMPS *); // constructor
        ~PairNonReciprocal() override;     // destructor

        void compute(int, int) override;   // required method - workhorse routine that computes pairwise interactions
        void settings(int, char **) override;       // required method - processes the arguments to the pair_style command (at the beginning)
        void coeff(int, char **) override;          // required method - set coefficients for one i,j type pair, called from pair_coeff
        void init_style() override;                 // optional method - style initialization: request neighbour list(s), error checks
        double init_one(int, int) override;

       protected:

        void   allocate(); // allocates dynamic memory(amount of memory needed depends on user - variables we introduce from outside)

        // variables of the potential
        double cut_global;     // global interaction range (same name as in LJ)
        double **cut;           // (same as in LJ)
        int    exponent;       // exponent for the force
        int    sigma_tilde;    // the distance between particles when they touch
        double int_scale;      // scale for the interaction
        double **activity;     // activity of the catalytic colloid
        double **mobility;     // mobility of the catalytic colloid
        double *cut_respa;
      };
    }
  #endif
#endif
