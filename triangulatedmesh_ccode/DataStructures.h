//
//  DataStructures.h
//  TriangulatedMonteCarloMembrane
//
//

#ifndef DataStructures_h
#define DataStructures_h

#include "PreprocessorDeclarations.h"
#include <stdio.h>

typedef struct{

  // Constraint inputs
  double pres;  // K_stretching of volume
  double gamma; // K_stretching of surface area
  double kappa; // bending constraint prefactor

  // Related to "box" side length
  double sidex,sidey,sidez;
  double Isidex,Isidey,Isidez;
  double side2x,side2y,side2z;

  double vol,Ivol,volC;
  int Lcelx,Lcely,Lcelz,Ncel;
  double celsidex,celsidey,Icelsidex,Icelsidey;
  double celsidez,Icelsidez;

  double etaC,etaR,etaR2;
  //double sig1[50000+1]; --> SEEMINGLY USELESS, commenting it out
  double sig2,sig1_sq,sum_sig,um_sig_sq,q;
  double rv,rc,rb,rv2,rb2,Vgap;
  double rb2_2D;
  double C0,C1,C2,CC;
  double eps_mc,eps_mv;

  double rad;
  double Svol;
  double SurfArea;
  double eqBondDist; // HARDCODED?

  double EbendChange;
  double EbindChange_colloid_bead;
  double EbindChange_bead_colloid;
  double Evolchange;
  double Esurfchange;

  // (deprecated)
  //double EstretchChange;
  //double springEnN;
  //double springEnO;
  //double springEnCh;

} SYSTEM;

typedef struct{
  double x,y,z;      /* Coordinates (x,y,z)         */
  double Vx,Vy,Vz;   /* Verlet Coordinates (x,y,z) for bead-bead interactions */
  double vx,vy,vz;   /* Verlet Coordinates (x,y,z) for colloid-bead interactions */
  double vcx, vcy, vcz; /* CONTACTS */
  int verl[2700],bond[2700],clus[2700];
  int list[9100+1];
  int nlist;
  int type;
  int Nverl,Nbond;
  int cellID;
  int before,after;
  int Ng,ng[15];
  int ngR[15];
  int ngL[15];

  int contact_list[9100+1];
  int ncontact;
} PARTICLE;

typedef struct{
  int Nv,v[3+1];
  int Nt,t[3+1];
} TRIANGLE;

typedef struct{
  int ngb[30];
  int begin;
} CELL;

typedef struct{
  double x,y;
} vec2D;

typedef struct{
  double x,y,z;
} vec3D;

typedef struct{
  double theta,phi;
} ANGOLO;

typedef struct{
  double x,y,z;      /* Coordinates (x,y,z)         */
} MEMBRANE;

typedef struct{
 double sig;
 double sig2;
 double rad;
 double dis_mc,rv,rv2,rc, rc2;
 double update_bcoll_list;
} COLL;

// ***************************************************
// GLOBAL STRUCTURES
// ***************************************************
SYSTEM S;
PARTICLE *Elem;
TRIANGLE *Tr;
MEMBRANE *MEM;
CELL *Cella;
COLL Coll;

double V0,A0;

// ***************************************************
// GLOBAL VARIABLES
// ***************************************************
int Ntri, N, Ncoll, counter, seed;
int Tmax=10; // [ON]: never allow more then Tmax-coordinated points.
int is_mem_fluid;
int start_from_config;
int L_system;
int ID_force;
int inorout;

double D0, phi1, phi2, theta1, theta2;
double lattice,lattice_y;
double cutoff,cutoff2;
double cm_phi,cm_theta;

// source of non-reciprocity
double factor_bead_colloid;
double factor_colloid_bead;
double internal_factor_bead_colloid;
double internal_factor_colloid_bead;
double k_harmonic = 1.0;
double r_eq = 22.0;

double factor_LJ;

// pulling equilibrium
double equilibrium;
double eq_old;

char readname[500];   // file name that contains initial conditions?
char errorfile[500];  // errorfile information
char Roname[500];     // file name for the density?
char Prob_name[500];

FILE *read,*wr1,*wr2,*prob;

vec3D Centre;

int time_reciprocal;
double factor_reciprocal;
int accepted_moves;
double acceptance_rate;

#endif /* DataStructures_h */
