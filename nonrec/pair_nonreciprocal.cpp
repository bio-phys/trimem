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
   [IF PROBLEMS WITH SIGNS] Additional information about the implementation
   of the potential can be obtained from R. Golestanian, "Phoretic active matter" (2019).

   - Whether neighbor lists are correct or not has been checked by implementing
   a LJ potential, and seing whether the trajectories are equal (see _SanityChecks folder)

------------------------------------------------------------------------- */

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>

#include "atom.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "comm.h"
#include "force.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "utils.h"


#include "pair_nonreciprocal.h"

using namespace LAMMPS_NS;
using namespace std;

/* ---------------------------------------------------------------------- */

PairNonReciprocal::PairNonReciprocal(LAMMPS *lmp) : Pair(lmp)
{
  /* Constructor of PairNonReciprocal class.
     According to the LAMMPS manual (p. 689),
     some flags must be defined, depending on
     what methods are defined in this .cpp file
     -- NOT ENTIRELY SURE WHAT FLAGS TO ENABLE --
     (I think this depends on the implementation)
  */
  restartinfo   = 0; // not implemented at the moment (must look in detail)

}

/* ---------------------------------------------------------------------- */

PairNonReciprocal::~PairNonReciprocal()
{
  /* Destructor of PairNonReciprocal class */
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(activity);
    memory->destroy(mobility);
  }
}

/* ---------------------------------------------------------------------- */

void PairNonReciprocal::compute(int eflag, int vflag)
{

  /* Main function for PairNonReciprocal
    Computes the pairwise, non-reciprocal interaction potential.
    - The magnitude of the force that particle 1 exerts on particle 2 is:
       F_12 = activity_1 * mobility_2 / (24*pi*diffusivity) * (1/r2)
    - The magnitude of the force that particle 2 exerts on particle 1 is:
       F_21 = activity_2 * mobility_1 / (24*pi*diffusivity) * (1/r2)
  */

  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double r2inv, r6inv, forcelj, factor_lj; // for lJ verifications
  double r,rsq,rinv,rexpinv; // part of the original non recp code
  int *ilist,*jlist,*numneigh,**firstneigh;

  // defined by me
  double catcoll_coeff_i, catcoll_coeff_j;
  double unitvec_x, unitvec_y, unitvec_z;
  double fxi, fyi, fzi, fxj, fyj, fzj;
  double vxi, vyi, vzi, vxj, vyj, vzj;

  evdwl = 0.0;
  ev_init(eflag, vflag); // eflag - energy computation, vflag - virial computation

  double **x = atom->x;
  double **f = atom->f;
  double **v = atom->v;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

  for (ii = 0; ii < inum; ii++) {
    i     = ilist[ii];
    xtmp  = x[i][0];
    ytmp  = x[i][1];
    ztmp  = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum  = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j         = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j        &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {

        /*
        // DIRECTLY COPIED FROM LAMMPS FOR LENNARD-JONES TO VERIFY NEIGHBOUR LISTS
        r2inv = 1.0 / rsq;
        r6inv = r2inv * r2inv * r2inv;
        forcelj = r6inv * (48.0*activity[itype][jtype] * r6inv - 24.0*activity[itype][jtype]);
        fpair = factor_lj * forcelj * r2inv;

        f[i][0] += delx * fpair;
        f[i][1] += dely * fpair;
        f[i][2] += delz * fpair;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx * fpair;
          f[j][1] -= dely * fpair;
          f[j][2] -= delz * fpair;
        }
        */

        // -------------------------------------------------
        //  MY NON-RECIPROCAL CODE
        // -------------------------------------------------

        r = sqrt(rsq);
        rinv  = 1/r;
        // keep in mind that in the force there would be one sigma tilde less
        // has to be absorbed by the activity or the mobility
        rexpinv = pow(sigma_tilde*rinv, exponent);

        unitvec_x = delx * rinv;
        unitvec_y = dely * rinv;
        unitvec_z = delz * rinv;

        // assuming diffusivity to be just one (first index dominates)
        catcoll_coeff_i = activity[itype][jtype] * mobility[jtype][itype] * int_scale;
        catcoll_coeff_j = activity[jtype][itype] * mobility[itype][jtype] * int_scale;

        // forces acting on i are due to field created by j (J ON I)
        fxi = catcoll_coeff_j * rexpinv * unitvec_x;
        fyi = catcoll_coeff_j * rexpinv * unitvec_y;
        fzi = catcoll_coeff_j * rexpinv * unitvec_z;

        // forces acting on j are due to field created by i (I ON J)
        fxj = - catcoll_coeff_i * rexpinv * unitvec_x;
        fyj = - catcoll_coeff_i * rexpinv * unitvec_y;
        fzj = - catcoll_coeff_i * rexpinv * unitvec_z;

        // update forces on i
        f[i][0] += fxi;
        f[i][1] += fyi;
        f[i][2] += fzi;

        if (newton_pair || j < nlocal) {
          // update forces on j
          f[j][0] += fxj;
          f[j][1] += fyj;
          f[j][2] += fzj;
        }


        // -------------------------------------------------
        //  MODIFYING PARTICLE VELOCITIES INSTEAD OF FORCES
        //  please note that the sign change is due to using a single vector
        //  to define the direction of motion. Signs agree with Golestanian paper.
        //  Problematic for keeping the temperature of the system?
        //  Or is it just that the temperature of the system is measured from the velocity
        //  in which case it is just fucked up because we are injecting energy here?
        // -------------------------------------------------
        /*
        r = sqrt(rsq);
        rinv  = 1.0/r;
        rexpinv = pow(rinv, 2);

        unitvec_x = delx * rinv;
        unitvec_y = dely * rinv;
        unitvec_z = delz * rinv;

        // the interaction scale = velocity scale, contains diffusivities of chemicals
        catcoll_coeff_i = activity[itype][jtype] * mobility[jtype][itype] * int_scale;
        catcoll_coeff_j = activity[jtype][itype] * mobility[itype][jtype] * int_scale;

        // velocity drift on i due to field created by j (J ON I)
        vxi = catcoll_coeff_j * rexpinv * unitvec_x;
        vyi = catcoll_coeff_j * rexpinv * unitvec_y;
        vzi = catcoll_coeff_j * rexpinv * unitvec_z;

        // forces acting on j are due to field created by i (I ON J)
        vxj = - catcoll_coeff_i * rexpinv * unitvec_x;
        vyj = - catcoll_coeff_i * rexpinv * unitvec_y;
        vzj = - catcoll_coeff_i * rexpinv * unitvec_z;

        // update velocity on i
        v[i][0] += vxi;
        v[i][1] += vyi;
        v[i][2] += vzi;

        // update velocity on j
        v[j][0] += vxj;
        v[j][1] += vyj;
        v[j][2] += vzj;
        */
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}

/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairNonReciprocal::allocate()
{

  /*
  For whatever obscure reason, one must have n+1 space to allocate things
  I think this is because users don't start particle labelling from 0
  I seem to be forced to allocate cutsq too?
  */

  allocated = 1;
  int n = atom->ntypes +1;

  memory->create(setflag, n, n, "pair:setflag");

  for (int i = 1; i < n; i++){
    for (int j = i; j < n; j++){
      setflag[i][j] = 0;
    }
  }

  memory->create(cutsq,    n, n, "pair:cutsq");
  memory->create(cut,      n, n, "pair:cut");
  memory->create(activity, n, n, "pair:activity");
  memory->create(mobility, n, n, "pair:mobility");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairNonReciprocal::settings(int narg, char **arg)
{
  /*
    NOT ENTIRELY SURE what this function does exactly
    although it seems it is reading coefficients one may
    define in the settings part (like the global cutoff)
    -- probably unnecessary to input global cutoff from outside
  */

  if (narg !=  4)
    error->all(FLERR,"Illegal pair_style command.");

  // global cutoff and interaction scale
  cut_global  = utils::numeric(FLERR,arg[0],true,lmp);
  int_scale   = utils::numeric(FLERR,arg[1],true,lmp);
  exponent    = utils::numeric(FLERR,arg[2],true,lmp);
  sigma_tilde = utils::numeric(FLERR,arg[3],true,lmp);

  // not sure what is going on here
  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++) {
      for (j = i; j <= atom->ntypes; j++) { // ------> point of change
        if (setflag[i][j]) {
          cut[i][j] = cut_global;
        }
      }
    }
  }
}


/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairNonReciprocal::coeff(int narg, char **arg)
{

  /*
  PAIR_STYLE needs 7 arguments: 2 for the particle IDs and
  (1) activity_i, (2) activity_j, (3) mobility_i, (4) mobility_j
  (5) cutoff (6) diffusivity
  */

  if (narg < 6 || narg > 8)
    error->all(FLERR,"Incorrect args for pair_style catcolloid coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  utils::bounds(FLERR,arg[0],1,atom->ntypes,ilo,ihi,error); // defines atom type i
  utils::bounds(FLERR,arg[1],1,atom->ntypes,jlo,jhi,error); // defines atom type j

  double activity_i      = utils::numeric(FLERR,arg[2],true,lmp);
  double activity_j      = utils::numeric(FLERR,arg[3],true,lmp);
  double mobility_i      = utils::numeric(FLERR,arg[4],true,lmp);
  double mobility_j      = utils::numeric(FLERR,arg[5],true,lmp);

  // optional (one can specify a cutoff here)
  double cut_one        = cut_global;
  if(narg == 7) cut_one = utils::numeric(FLERR, arg[6], true, lmp);

  // filling-in particle properties
  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      activity[i][j] = activity_i;
      activity[j][i] = activity_j;
      mobility[i][j] = mobility_i;
      mobility[j][i] = mobility_j;
      cut[i][j]      = cut_one;
      cut[j][i]      = cut_one;
      setflag[i][j]  = 1;
      setflag[j][i]  = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");

  /*
  std::cout << "Activities :: " << std::endl;
  for (int i = ilo; i <= ihi; i++) {
    std::cout << "Type " << i << " activity = " << activity[i] << std::endl;
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      std::cout << "Type " << j << " activity = " << activity[j] << std::endl;
    }
  }

  std::cout << "Mobilities :: " << std::endl;
  for (int i = ilo; i <= ihi; i++) {
    std::cout << "Type " << i << " mobility = " << mobility[i] << std::endl;
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      std::cout << "Type " << j << " mobility = " << mobility[j] << std::endl;
    }
  }

  std::cout << "diffusivity :: " << std::endl;
  for (int i = ilo; i <= ihi; i++) {
    std::cout << "Type " << i << " diffusivity = " << diffusivity[i] << std::endl;
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      std::cout << "Type " << j << " diffusivity = " <<diffusivity[j] << std::endl;
    }
  }
  */

  /*std::cout << "This is a test to see what happens to the cutoff" << std::endl;
  std::cout << "Element 1 1 " << cutoff[1][1] << std::endl;
  std::cout << "Element 1 2 " << cutoff[1][2] << std::endl;
  std::cout << "Element 2 1 " << cutoff[2][1] << std::endl;
  std::cout << "Element 2 2 " << cutoff[2][2] << std::endl;
  std::cout << "Given that I am only defining cross elements, I do not want" << std::endl;
  std::cout << "particles in the diagonal to be able to interact." << std::endl;
  */

  /*std::cout << "This is a test to check activities" << std::endl;
  std::cout << "Element 1 1 " << activity[1][1] << std::endl;
  std::cout << "Element 1 2 " << activity[1][2] << std::endl;
  std::cout << "Element 2 1 " << activity[2][1] << std::endl;
  std::cout << "Element 2 2 " << activity[2][2] << std::endl;

  std::cout << "This is a test to check mobilities" << std::endl;
  std::cout << "Element 1 1 " << mobility[1][1] << std::endl;
  std::cout << "Element 1 2 " << mobility[1][2] << std::endl;
  std::cout << "Element 2 1 " << mobility[2][1] << std::endl;
  std::cout << "Element 2 2 " << mobility[2][2] << std::endl;*/
}

/* ----------------------------------------------------------------------
   init specific to this pair style
------------------------------------------------------------------------- */

void PairNonReciprocal::init_style()
{
  auto req = neighbor->add_request(this);
}

double PairNonReciprocal::init_one(int i, int j){

  cut[j][i] = cut[i][j];

  return cut[i][j];
}
