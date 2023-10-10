// clang-format off
/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   https://www.lammps.org/, Sandia National Laboratories
   LAMMPS development team: developers@lammps.org

   This software is distributed under the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Axel Kohlmeyer (Temple U)
------------------------------------------------------------------------- */

#include "pair_nonreciprocal_omp.h"

#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <iostream>




#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "suffix.h"
#include "memory.h"
#include "error.h"
#include "update.h"
#include "utils.h"
#include "omp_compat.h"




using namespace LAMMPS_NS;

/* ---------------------------------------------------------------------- */

PairNonReciprocalOMP::PairNonReciprocalOMP(LAMMPS *lmp) :
  PairNonReciprocal(lmp), ThrOMP(lmp, THR_PAIR)
{
  suffix_flag |= Suffix::OMP;
  respa_enable = 0;
  cut_respa = nullptr;
}

/* ---------------------------------------------------------------------- */

void PairNonReciprocalOMP::compute(int eflag, int vflag)
{
  ev_init(eflag,vflag);

  const int nall = atom->nlocal + atom->nghost;
  const int nthreads = comm->nthreads;
  const int inum = list->inum;

#if defined(_OPENMP)
#pragma omp parallel LMP_DEFAULT_NONE LMP_SHARED(eflag,vflag)
#endif
  {
    int ifrom, ito, tid;

    loop_setup_thr(ifrom, ito, tid, inum, nthreads);
    ThrData *thr = fix->get_thr(tid);
    thr->timer(Timer::START);
    ev_setup_thr(eflag, vflag, nall, eatom, vatom, nullptr, thr);

    if (evflag) {
      if (eflag) {
        if (force->newton_pair) eval<1,1,1>(ifrom, ito, thr);
        else eval<1,1,0>(ifrom, ito, thr);
      } else {
        if (force->newton_pair) eval<1,0,1>(ifrom, ito, thr);
        else eval<1,0,0>(ifrom, ito, thr);
      }
    } else {
      if (force->newton_pair) eval<0,0,1>(ifrom, ito, thr);
      else eval<0,0,0>(ifrom, ito, thr);
    }
    thr->timer(Timer::PAIR);
    reduce_thr(this, eflag, vflag, thr);
  } // end of omp parallel region
}

template <int EVFLAG, int EFLAG, int NEWTON_PAIR>
void PairNonReciprocalOMP::eval(int iifrom, int iito, ThrData * const thr)
{
  const auto * _noalias const x = (dbl3_t *) atom->x[0];
  auto * _noalias const f = (dbl3_t *) thr->get_f()[0];
  const int * _noalias const type = atom->type;
  const double * _noalias const special_lj = force->special_lj;
  const int * _noalias const ilist = list->ilist;
  const int * _noalias const numneigh = list->numneigh;
  const int * const * const firstneigh = list->firstneigh;


  double xtmp,ytmp,ztmp,delx,dely,delz,fxtmp,fytmp,fztmp;
  double rsq,r2inv,r6inv,forcelj,factor_lj,evdwl,fpair;

  #
  double r,rinv,rexpinv,r2invs;
  #

  const int nlocal = atom->nlocal;
  int j,jj,jnum,jtype,halfexponent,modexponent,ie;

  evdwl = 0.0;
  halfexponent = static_cast<int>(std::floor(exponent / 2));
  modexponent = std::fmod(exponent, 2);






  // loop over neighbors of my atoms

  for (int ii = iifrom; ii < iito; ++ii) {
    const int i = ilist[ii];
    const int itype = type[i];
    const int    * _noalias const jlist = firstneigh[i];
    const double * _noalias const cutsqi = cutsq[itype];



    #
    double catcoll_coeff_i, catcoll_coeff_j;

    double fxi, fyi, fzi, fxj, fyj, fzj;
    #

    xtmp = x[i].x;
    ytmp = x[i].y;
    ztmp = x[i].z;
    jnum = numneigh[i];
    fxtmp=fytmp=fztmp=0.0;

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j].x;
      dely = ytmp - x[j].y;
      delz = ztmp - x[j].z;
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsqi[jtype]) {

        //const double  activityij = ;
        //const double  activityij = ;
        //const double  activityji = activity[jtype][itype];
        //const double  mobilityij = mobility[itype][jtype];
        //const double  mobilityji = mobility[jtype][itype];


        r2inv = 1.0/rsq;
        r2invs = r2inv*sigma_tilde*sigma_tilde;
        rexpinv=1.0;

        // unitvector already incorporated in rexpinv
        for (ie = 0;ie<halfexponent;ie++) {
        rexpinv *= r2invs;
        }
        if ( modexponent == 0) {
        r = sqrt(rsq);
        rinv=1.0/r;
        rexpinv *= rinv;
        }
        else{
        rexpinv *= sigma_tilde*r2inv;
        }






        // assuming diffusivity to be just one (first index dominates)
        catcoll_coeff_i = activity[itype][jtype] * mobility[jtype][itype] * int_scale;
        catcoll_coeff_j = activity[jtype][itype] * mobility[itype][jtype] * int_scale;

        // forces acting on i are due to field created by j (J ON I)
        fxi = catcoll_coeff_j * rexpinv * delx;
        fyi = catcoll_coeff_j * rexpinv * dely;
        fzi = catcoll_coeff_j * rexpinv * delz;

        // forces acting on j are due to field created by i (I ON J)
        fxj = - catcoll_coeff_i * rexpinv * delx;
        fyj = - catcoll_coeff_i * rexpinv * dely;
        fzj = - catcoll_coeff_i * rexpinv * delz;

        fxtmp += fxi;
        fytmp += fyi;
        fztmp += fzi;





        if (NEWTON_PAIR || j < nlocal) {
          f[j].x += fxj;
          f[j].y += fyj;
          f[j].z += fzj;
        }

        if (EFLAG) {
          evdwl = 0;

        }

        if (EVFLAG) ev_tally_thr(this,i,j,nlocal,NEWTON_PAIR,
                                 evdwl,0.0,0.0,delx,dely,delz,thr);
      }
    }
    f[i].x += fxtmp;
    f[i].y += fytmp;
    f[i].z += fztmp;


  }
}

/* ---------------------------------------------------------------------- */

double PairNonReciprocalOMP::memory_usage()
{
  double bytes = memory_usage_thr();
  bytes += PairNonReciprocal::memory_usage();

  return bytes;
}
