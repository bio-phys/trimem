################################################################################
# TriLmp: TRIMEM (Siggel et. al, 2023) + LAMMPS                                #
# Enabling versatile MD simulations w/dynamically triangulated membranes       #
#                                                                              #
# This program was originally created during the summer of 2023                #
# by Michael Wassermair during an internship in the Saric Group (ISTA).        #
# It is based on TRIMEM (Siggel et al.), which was originally intended         #
# to perform Hybrid Monte Carlo (HMC) energy minimization of the Helfrich      #
# Hamiltonian using a vertex-averaged triangulated mesh discretisation.        #
# By connecting the latter with LAMMPS, we expose the mesh vertices            #
# (pseudo-particles). This allows us to perform simulations of a               #
# triangulated membrane under different scenarios, detailed below.             #                                                
#                                                                              #
# The program depends on a modified version of trimem and LAMMPS that uses     #
# specific packages and the additional pair_styles nonreciprocal               #
# and nonreciprocal/omp (see SETUP_GUIDE.txt and HowTo_TriLMP.md for details)  #
#                                                                              #
# ORIGINAL PROGRAM STRUCTURE                                                   #
# - Internal classes used by TriLmp                                            #
# - TriLmp Object                                                              #
#                                                                              #
# --- Default Parameters -> description of all parameters                      #
# --- Initialization of Internal classes using parameters                      #
# --- Init. of TRIMEM EnergyManager used for gradient/energy                   #
#     of helfrich hamiltonian                                                  #
#                                                                              #
# --- LAMMPS initialisation                                                    #
# ---+++ Creating instances of lmp                                             #
# ---+++ Setting up Basic system                                               #
# ---+++ Calling Lammps functions to set thermostat and interactions           #
#                                                                              #
# --- FLIPPING Functions                                                       #
# --- HMC/MD Functions + Wrapper Functions used on TRIMEM side                 #
# --- RUN Functions -> simulation utility to be used in script                 #
# --- CALLBACK and OUTPUT                                                      #
# --- Some Utility Functions                                                   #
# --- Minimize Function -> GD for preconditionining states for HMC             #
# --- Pickle + Checkpoint Utility                                              #
# --- LAMMPS scrips used for setup                                             #
#                                                                              #
# The code has been restructured and rewritten by Maitane Munoz Basagoiti      #
# after its initial implementation. The TriLMP class has been reduced to its   #
# minimum with the goal of having a more organic coupling to LAMMPS. The       #
# main motivation of this rewriting was to improve code readability, make      #
# debugging easier and simplify the process of adding features to the          #
# simulations.                                                                 #
#                                                                              #
# The code is stored in a GitHub repository. It is therefore subjected to      #
# version control. Should you be interested in checking older versions,        #
# please refer to that.                                                        #
#                                                                              #             
# CURRENT TRILMP FUNCTIONALITIES                                               #
# 1. Triangulated vesicle in solution                                          #
#   1.1. Tube pulling experiment                                               #
# 2. Triangulated vesicle in the presence of                                   #
#   2.1. Nanoparticles/symbionts                                               #
#   2.2. Elastic membrane (S-layer)                                            #
#   2.3. Rigid bodies/symbionts                                                #
# 3. Extended simulation functionalities                                       #
#   3.1. GCMC Simulations                                                      #
#   3.2. Chemical reactions                                                    #
#                                                                              #
# The code is currently mantained by the following members of the Saric group  #
# (Please write your name + email in the lines below if it applies)            #
# - Maitane Munoz Basagoiti (MMB) - maitane.munoz-basagoiti@ista.ac.at         #
# - name + email                                                               #
#                                                                              #
# Implementation codes of chemical reactions without topology information:     #
# - Felix Wodaczek (FW) - felix.wodaczek@ista.ac.at                            #                                
#                                                                              #
# ADDITIONAL NOTES                                                             #
# To perform HMC simulation that accepts all configurations:                   #
#   - thermal_velocities=False, pure_MD=True                                   #
# To perform HMC simulation as in Trimem:                                      #
#   - thermal_velocities=True, pure_MD=False)                                  #
#                                                                              #
# NOTE - ABOUT ADDING NEW ATTRIBUTES TO THE CONSTRUCTOR OF TRILMP              #
# - If you want to extend the functionalities of the TriLMP class by adding    #
#   new attributes to the constructor, please make sure you also include them  #
#   in the __reduce__ method so that the pickling of the object can be done    #
#   can be done correctly. The order in which arguments are passed to the      #
#   __reduce__ method seems to matter.                                         #
################################################################################

################################################################################
#                             IMPORTS                                          #
################################################################################

import numpy as np
import trimem.mc.trilmp_h5
import trimesh, re, textwrap, warnings, psutil, os, sys, time, pickle, pathlib, json

from copy import copy
from ctypes import *
from .. import core as m
from trimem.core import TriMesh
from trimem.mc.mesh import Mesh
from collections import Counter
from scipy.optimize import minimize
from trimem.mc.output import make_output
from typing import Union as pyUnion # to avoid confusion with ctypes.Union
from datetime import datetime, timedelta
from lammps import lammps, PyLammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM, LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY, LMP_SIZE_VECTOR, LMP_SIZE_ROWS, LMP_SIZE_COLS

_sp = u'\U0001f604'
_nl = '\n'+_sp

# functions used to write blocks of text
def block(s):
    return _nl.join(s.split('\n'))

def dedent(s):
    ls = s.split('\n')
    if ls[-1]=='':
        ls.pop()
    c = None
    for i,l in enumerate(ls):
        if l and l[0]==_sp:
            assert(i!=0)
            if c==None:
                c_g=re.match(r"\s*", ls[i-1])
                assert c_g is not None
                c = c_g.group()
            ls[i]=c+ls[i][1:]
        else:
            c=None
    return textwrap.dedent('\n'.join(ls))

################################################################################
#                   INTERNAL CLASSES TO BE USED BY TriLmP                      #
################################################################################

class Timer():
    """

    Storage for timer state to reinitialize PerformanceEnergyEvaluator after Reset
    
    """
    def __init__(self,ptime,ts,ta,tan,ts_default,stime):
        self.performance_start=ptime
        self.performance_timestamps=ts
        self.timearray=ta
        self.timearray_new=tan
        self.timestamps=ts_default
        self.start=stime

class InitialState():
    def __init__(self,area,volume,curvature,bending,tethering):
        """
        
        Storage for reference properties to reinitialize Estore after Reset. In case of reinstating Surface
        Repulsion in TRIMEM a repulsion property would have to be added again 
        
        """
        self.area=area
        self.volume=volume
        self.curvature=curvature
        self.bending=bending
        self.tethering=tethering

class Beads():
    def __init__(self,n_beads, n_bead_types,bead_pos, bead_v, bead_sizes,bead_types):
        """
        
        Storage for external bead parameters
        Args:
            n_beads      : number of external beads in the system
            n_bead_types : number of different types of beads used
            bead_pos     : (n_beads,3) array of bead positions
            bead_v       : (n_beads, 3) array of bead velocities
            bead_sizes   : (n_bead_types,1) tuple containing the sizes of the beads, e.g. (size_bead1) or (size_bead1,size_bead2)
            bead_types   : (n_beads,1) tuple or array (must use 1 index) of the different types of the beads.
                Bead types are strictly >=2, e.g. 3 beads and 2 bead_types (2,3,3)
            args: ignored

        Keyword Args:
            kwargs: ignored

        """
        self.n_beads=n_beads
        self.n_bead_types=n_bead_types
        self.positions=bead_pos
        self.velocities = bead_v
        self.types=bead_types
        self.bead_sizes=bead_sizes  

class OutputParams():
    """
    
    Containter for parameters related to the output option 
    
    """
    def __init__(self,
                 info,
                 thin,
                 out_every,
                 input_set,  # hast to be stl file or if None uses mesh
                 output_prefix,
                 restart_prefix,
                 checkpoint_every,
                 output_format,
                 output_flag,
                 output_counter,
                 performance_increment,
                 energy_increment
                 ):
        self.info=info
        self.thin=thin
        self.out_every=out_every
        self.input_set=input_set
        self.output_prefix=output_prefix
        self.restart_prefix=restart_prefix
        self.checkpoint_every = checkpoint_every
        self.output_format = output_format
        self.output_flag = output_flag
        self.output_counter=output_counter
        self.performance_increment=performance_increment
        self.energy_increment=energy_increment

class AlgoParams():
    """
    
    Containter for parameters related to the algorithms used  -> See DEFAULT PARAMETERS section for description
    
    """
    def __init__(self,
                 num_steps,
                 reinitialize_every,
                 init_step,
                 step_size,
                 traj_steps,
                 momentum_variance,
                 flip_ratio,
                 flip_type,
                 initial_temperature,
                 cooling_factor,
                 start_cooling,
                 maxiter,
                 refresh,
                 thermal_velocities,
                 pure_MD,
                 switch_mode,
                 box
                 ):

        self.num_steps=num_steps
        self.reinitialize_every=reinitialize_every
        self.init_step=init_step
        self.step_size=step_size
        self.traj_steps=traj_steps
        self.momentum_variance=momentum_variance
        self.flip_ratio=flip_ratio
        self.flip_type=flip_type
        self.initial_temperature=initial_temperature
        self.cooling_factor=cooling_factor
        self.start_cooling=start_cooling
        self.maxiter=maxiter
        self.refresh=refresh
        self.thermal_velocities=thermal_velocities
        self.pure_MD=pure_MD
        self.switch_mode=switch_mode
        self.box=box

################################################################################
#                  MAIN TRILMP CLASS OBJECT                                    #
################################################################################

class TriLmp():

    def __init__(self,

                 # INITIALIZATION
                 initialize=True,                  # determines if mesh is used as new reference in estore
                 debug_mode=True,                  # determines how much of the LAMMPS output will be printed
                 num_particle_types=1,             # how many particle types will there be in the system
                 mass_particle_type=[1],           # the mass of the particle per type
                 group_particle_type=['vertices'], # insert the name of each group

                 # TRIANGULATED MEMBRANE MESH
                 mesh_points=None,    # positions of membrane vertices
                 mesh_faces=None,     # faces defining mesh
                 mesh_velocity=None,  # initial velocities
                 vertices_at_edge=None, # vertices at the edge of a flat membrane

                 # TRIANGULATED MEMBRANE BONDS
                 bond_type='Edge',      # 'Edge' or 'Area
                 bond_r=2,              # steepness of potential walls (from Trimem)
                 lc0=None,              # upper onset ('Edge') default will be set below
                 lc1=None,              # lower onset ('Edge')
                 a0=None,               # reference face area ('Area')

                 # TRIANGULATED MEMBRANE MECHANICAL PROPERTIES
                 n_search="cell_list",       # neighbour list types ('cell and verlet') -> NOT USED
                 rlist=0.1,                  # neighbour list cutoff -> NOT USED
                 exclusion_level=2,          # neighbourhood exclusion setting -> NOT IMPLEMENTED
                 rep_lc1=None,               # lower onset for surface repusion (default set below)
                 rep_r= 2,                   # steepness for repulsion potential
                 delta= 0.0,                 # "timestep" parameter continuation for hamiltonian
                 lam= 1.0,                   # cross-over parameter continuation for hamiltonian (see trimem -> NOT FUNCTIONAL)
                 kappa_b = 20.0,             # bending modulus
                 kappa_a = 1.0e6,            # area penalty
                 kappa_v = 1.0e6,            # volume penalty
                 kappa_c = 1.0e6,            # curvature (area-difference) penalty
                 kappa_t = 1.0e5,            # tethering
                 kappa_r = 1.0e3,            # surface repulsion
                 area_frac = 1.0,            # target area
                 volume_frac = 1.0,          # target volume
                 curvature_frac = 1.0,       # target curvature
                 print_bending_energy=True,  # might increase simulation run - think of putting it as false
                 # PROGRAM PARAMETERS
                 num_steps=10,                        # number of overall simulation steps (for trilmp.run() but overitten by trilmp.run(N))
                 reinitialize_every=10000,            # NOT USED - TODO
                 init_step='{}',                      # NOT USED - TODO
                 step_size=7e-5,                      # MD time step
                 traj_steps=100,                      # MD trajectory length
                 momentum_variance=1.0,               # mass of membrane vertices
                 flip_ratio=0.1,                      # fraction of flips intended to flip
                 flip_type='parallel',                # 'serial' or 'parallel'
                 initial_temperature=1.0,             # temperature of system
                 cooling_factor=1.0e-4,               # cooling factor for simulated anneadling -> NOT IMPLEMENTED
                 start_cooling=0,                     # sim step at which cooling starts -> NOT IMPLEMENTED
                 maxiter=10,                          # parameter used for minimize function (maximum gradient steps)
                 refresh=1,                           # refresh rate of neighbour list -> NOT USED TODO
                 thermal_velocities=False,            # thermal reset of velocities at the begin of each MD step
                 pure_MD=True,                        # if true, accept every MD trajectory
                 switch_mode='random',                # 'random' or 'alternating': sequence of MD, MC stages
                 box=(-100,100,-100,100,-100,100),    # simulation box
                 periodic=False,                      # periodic boundary conditions
                 equilibrated=False,                  # equilibration state of membrane
                 equilibration_rounds=-1,             # number of equilibration rounds

                 # OUTPUT FILE PARAMETERS
                 info=10,                     # output hmc and flip info to shell every nth step
                 thin=10,                     # output trajectory every nth program (MD + HC) step
                 out_every= 0,                # output minimize state every nth step (only for gradient minimization)
                 input_set='inp.stl',         # hast to be stl file or if True uses mesh -> NOT USED
                 output_prefix='inp',         # name of simulation files
                 restart_prefix='inp',        # name for checkpoint files
                 checkpoint_every= 1,         # checkpoint every nth step
                 output_format='xyz',         # output format for trajectory -> NOT YET (STATISFYINGLY) IMPLEMENTED
                 output_flag='A',             # initial flag for output (alternating A/B)
                 output_counter=0,            # used to initialize (outputted) trajectory number in writer classes
                 performance_increment=10,    # print performance stats every nth step to output_prefix_performance.dat
                 energy_increment=10,         # print total energy to energy.dat

                 # REFERENCE STATE DATA placeholders to be filled by reference state parameters (used to reinitialize estore)
                 area=1.0,          # reference area
                 volume=1.0,        # reference volume
                 curvature=1.0,     # reference curvature
                 bending=1.0,       # reference bending
                 tethering=1.0,     # reference tethering
                 #repulsion=1.0,    # would be used if surface repulsion was handeld by TRIMEM

                 # TIMING UTILITY (used to time performance stats)
                 ptime=time.time(),
                 ptimestamp=[],
                 dtimestamp=[],
                 timearray=np.zeros(2),
                 timearray_new=np.zeros(2),
                 stime=datetime.now(),

                 # COUNTERS
                 move_count=0,   # move counts (to initialize with)
                 flip_count=0,   # flip counts (to initialize with) -> use this setting to set a start step (move+flip)

                 # BEADS/NANOPARTICLES
                 # (must be initialized in init because affects input file)
                 n_beads=0,                     # number of nanoparticles
                 n_bead_types=0,                # number of nanoparticle types
                 bead_pos=[],      # positioning of the beads
                 bead_v  =None,
                 bead_sizes=0.0,                # bead sizes
                 bead_types=[2],                 # bead types
                 n_bond_types = 1,

                 # EXTENSIONS MMB: ELASTIC MEMBRANE/S-LAYER
                 # (must be initialized in init because affects input file)
                 slayer = False,             # set true if simulation also contains elastic membrane
                 slayer_points=[],           # positions elastic membrane/S-layer vertices
                 slayer_bonds = [],          # bonds in elastic membrane/S-layer
                 slayer_dihedrals = [],      # dihedrals in elastic membrane/S-layer
                 slayer_kbond=1,             # bond harmonic stiffness
                 slayer_kdihedral=1,         # dihedral harmonic stiffness
                 slayer_rlbond=2**(1.0/6.0), # rest length of the bonds

                 # EXTENSIONS MMB: RIGID SYMBIONT
                 # (must be initialized in init because affects input file)
                 fix_rigid_symbiont=False,              # whether or not there will be a rigid symbiont
                 fix_rigid_symbiont_coordinates=None,   # coordinates that define the rigid symbiont
                 fix_rigid_symbiont_interaction=None,   # interactions of the rigid symbiont
                 fix_rigid_symbiont_params=None,        # parameters for the membrane-rigid symbiont interaction

                 # EXTENSIONS MMB: MIXED/HETEROGENEOUS MEMBRANES
                 # (must be initialized in init because affects input file)
                 heterogeneous_membrane=False,
                 heterogeneous_membrane_id = None,
                 
                 # TEST MODE FOR OPTIMIZATIONS 
                 test_mode = True,

                 ):


        ########################################################################
        #                            SOME MINOR PREREQUESITES                  #
        ########################################################################

        # initialization of (some) object attributes
        self.initialize           = initialize
        self.debug_mode           = debug_mode
        self.num_particle_types   = num_particle_types
        self.mass_particle_type   = mass_particle_type
        self.group_particle_type  = group_particle_type
        self.test_mode            = test_mode
        self.equilibrated         = equilibrated
        self.equilibration_rounds = equilibration_rounds
        self.print_bending_energy = print_bending_energy
        self.periodic             = periodic
        self.traj_steps           = traj_steps
        self.n_bond_types         = n_bond_types
        self.MDsteps              = 0.0
        self.acceptance_rate      = 0.0

        # used for (TRIMEM) minimization
        self.flatten = True
        if self.flatten:
            self._ravel = lambda x: np.ravel(x)
        else:
            self._ravel = lambda x: x

        # different bond types used for tether potential in TRIMEM
        self.bond_enums = {
            "Edge": m.BondType.Edge,
            "Area": m.BondType.Area
        }

        ########################################################################
        #                         MESH INITIALIZATION                          #
        # Triangulated mesh used in simulations must be a TriMEM 'Mesh' object #
        ########################################################################

        self.mesh_points = mesh_points
        self.mesh_faces = mesh_faces
        self.mesh_velocity=mesh_velocity
        self.vertices_at_edge=np.array(vertices_at_edge)

        # [from TriMem doc] class trimem.mc.mesh.Mesh(points=None, cells=None)
        self.mesh = Mesh(points=self.mesh_points, cells=self.mesh_faces)

        if pure_MD:
            self.mesh_temp=0
        else:
            self.mesh_temp = Mesh(points=self.mesh_points, cells=self.mesh_faces)

        # extract number of membrane vertices in simulation
        self.n_vertices=self.mesh.x.shape[0]
        # extract the number of membrane faces in simulation
        self.n_faces   = self.mesh.f.shape[0]

        ########################################################################
        #                       MESH BONDS/TETHERS                             #
        ########################################################################

        # initialize bond class 
        #[BondParams = class in the trimem.core module]
        #[BondType   = class in the trimem.core module]
        self.bparams = m.BondParams()
        if issubclass(type(bond_type), str):
            self.bparams.type = self.bond_enums[bond_type]
        else:
            self.bparams.type=bond_type

        # introduce steepness for bond potentials
        self.bparams.r = bond_r

        # MMB hardcoded scaling of the edges
        a, l = m.avg_tri_props(self.mesh.trimesh)
        self.bparams.lc0 = np.sqrt(3) # MMB: HARDCODED AS IN THE IN-HOUSE MC CODE
        self.bparams.lc1 = 1.0        # MMB: HARDCODED AS IN THE IN-HOUSE MC CODE
        self.bparams.a0 = a

        ########################################################################
        #          INITIALIZATION OF SUPERFACE REPULSION FOR TRIMEM            #
        # The code below is not used at the moment because TriLMP deals with   #
        # surface repulsion using a LAMMPS table.                              #
        ########################################################################

        # [from SurfaceRepulsionParams class in trimem.core]
        self.rparams = m.SurfaceRepulsionParams()

        # neighbour lists of TRIMEM are not used but kept here in case needed
        self.rparams.n_search = n_search
        self.rparams.rlist = rlist

        #currently default 2 is fixed, not yet implemented to change TODO
        self.rparams.exclusion_level = exclusion_level

        if rep_lc1==None:
            #by default set to average distance used for scaling tether potential
            self.rparams.lc1 = 1.0 # MMB: HARDCODED AS IN THE IN-HOUSE MC CODE
        else:
            self.rparams.lc1 = rep_lc1

        # surface repulsion steepness
        self.rparams.r = rep_r

        ########################################################################
        #   PARAMETER CONTINUATION: translate energy params (see TRIMEM)       #
        ########################################################################

        self.cp = m.ContinuationParams()
        self.cp.delta = delta
        self.cp.lam = lam

        ########################################################################
        #                           ENERGY PARAMETERS                          #
        # CAVEAT!! If one wants to change parameters between                   #
        # two trilmp.run() commands in a simulation script                     #
        # one has to alter trilmp.estore.eparams.(PARAMETER OF CHOICE).        #
        # These parameters will also be used to reinitialize LAMMPS, i.e       #
        # in pickling for the checkpoints                                      #
        ########################################################################

        self.eparams = m.EnergyParams()
        self.eparams.kappa_b = kappa_b
        self.eparams.kappa_a = kappa_a
        self.eparams.kappa_v = kappa_v
        self.eparams.kappa_c = kappa_c
        self.eparams.kappa_t = kappa_t
        self.eparams.kappa_r = kappa_r
        self.eparams.area_frac = area_frac
        self.eparams.volume_frac = volume_frac
        self.eparams.curvature_frac = curvature_frac
        self.eparams.bond_params = self.bparams
        self.eparams.repulse_params = self.rparams
        self.eparams.continuation_params = self.cp

        ########################################################################
        #                        ALGORITHMIC PARAMETERS                        #
        ########################################################################

        self.algo_params=AlgoParams(num_steps,reinitialize_every,init_step,step_size,traj_steps,
                 momentum_variance,flip_ratio,flip_type,initial_temperature,
                 cooling_factor,start_cooling,maxiter,refresh,thermal_velocities,pure_MD,switch_mode,box)

        ########################################################################
        #                      OUTPUT PARAMETERS                               #
        ########################################################################

        self.output_params=OutputParams(info,
                 thin,
                 out_every,
                 input_set,
                 output_prefix,
                 restart_prefix,
                 checkpoint_every,
                 output_format,
                 output_flag,
                 output_counter,
                 performance_increment,
                 energy_increment)

        ########################################################################
        #                ENERGY MANAGE INITIALIZATION WITH TRIMEM              #
        # From the TriMEM documentation:                                       #
        # - The 'trimem.core.EnergyManager' manages a particular parametri-    #
        #   zation of the Helfrich functional with additional penalties and    #
        #   tether-regularization.                                             #
        ########################################################################

        if self.initialize:
            # setup energy manager with initial mesh [class trimem.core.EnergyManager]
            self.estore = m.EnergyManagerNSR(self.mesh.trimesh, self.eparams)

            #save initial states property
            self.initial_state=InitialState(self.estore.initial_props.area,
                                            self.estore.initial_props.volume,
                                            self.estore.initial_props.curvature,
                                            self.estore.initial_props.bending,
                                            self.estore.initial_props.tethering)
                                            #self.estore.initial_props.repulsion)

            self.initialize=False

        else:
            # reinitialize using saved initial state properties (for reference potential V, A, dA)
            # the 'area', 'volume', 'curvature', 'bending' and 'thetering' properties must have been
            # saved by the pickle checkpoint; this allows you to restart the simulation with the same
            # specific properties
            self.initial_state = InitialState(area,
                                              volume,
                                              curvature,
                                              bending,
                                              tethering
                                              )
            self.init_props = m.VertexPropertiesNSR()
            self.init_props.area = self.initial_state.area
            self.init_props.volume = self.initial_state.volume
            self.init_props.curvature = self.initial_state.curvature
            self.init_props.bending = self.initial_state.bending
            self.init_props.tethering = self.initial_state.tethering

            #print(f"PASSING AREA: {area} INITIAL STATE AREA: {self.initial_state.area} INITIAL PROPS AREA: {self.initial_props.area}")
            #print(f"PASSING VOLUME: {volume} INITIAL STATE VOLUME: {self.initial_state.volume} INITIAL PROPS VOLUME: {self.initial_props.volume}")
            #print(f"PASSING CURVATURE: {curvature} INITIAL STATE CURVATURE: {self.initial_state.curvature} INITIAL PROPS CURVATURE: {self.initial_props.curvature}")
            #print(f"PASSING BENDING: {bending} INITIAL STATE BENDING: {self.initial_state.bending} INITIAL PROPS BENDING: {self.initial_props.bending}")
            #print(f"PASSING TETHERING: {tethering} INITIAL STATE BENDING: {self.initial_state.tethering} INITIAL PROPS TETHERING: {self.initial_props.tethering}")

            #recreate energy manager
            self.estore = m.EnergyManagerNSR(self.mesh.trimesh, self.eparams, self.init_props)

        # save general lengthscale, i.e. membrane bead "size" defined by tether repulsion onset
        self.l0 = 1.0 # MMB HARDCODED: LENGTH SCALE OF SYSTEM IS ALWAYS SIGMA = 1

        ########################################################################
        #                           BEADS/NANOPARTICLES                        #
        ########################################################################

        if len(bead_pos)>0:
            n_beads = len(bead_pos)

        self.beads=Beads(n_beads,
                         n_bead_types,
                         bead_pos,
                         bead_v,
                         bead_sizes,
                         bead_types)

        ########################################################################
        #                             EXTENSION: TETHERS                       #
        ########################################################################

        bond_text="""
                    special_bonds lj/coul 0.0 0.0 0.0
                    bond_style zero nocoeff
                    bond_coeff * * 0.0  """

        n_tethers=0
        add_tether=False

        ########################################################################
        #             EXTENSION: ELASTIC MEMBRANE/SLAYER                       #
        ########################################################################

        self.n_slayer           = 0
        self.slayer             = slayer
        self.slayer_points      = slayer_points
        self.slayer_bonds       = slayer_bonds
        self.slayer_dihedrals   = slayer_dihedrals
        self.slayer_kbond       = slayer_kbond
        self.slayer_rlbond      = slayer_rlbond
        self.slayer_kdihedral   = slayer_kdihedral
        self.n_slayer_dihedrals = 0
        self.n_slayer_bonds     = 0
        self.slayer_flag        = 0

        if self.slayer:
            self.n_slayer           = len(self.slayer_points)
            self.n_slayer_dihedrals = len(self.slayer_dihedrals)
            self.n_slayer_bonds     = len(self.slayer_bonds)
            self.slayer_flag        = 1
            n_bond_types+=1
            bond_text=f"""
                        special_bonds lj/coul 0.0 0.0 0.0
                        bond_style hybrid zero nocoeff harmonic
                        bond_coeff 1 zero 0.0
                        bond_coeff 2 harmonic {self.slayer_kbond} {self.slayer_rlbond}
                        dihedral_style harmonic
                        dihedral_coeff 1 {self.slayer_kdihedral} 1 1
                    """

        ########################################################################
        #                   EXTENSION: RIGID SYMBIONT                          #
        ########################################################################

        self.fix_rigid_symbiont             = fix_rigid_symbiont
        self.fix_rigid_symbiont_coordinates = fix_rigid_symbiont_coordinates
        self.fix_rigid_symbiont_nparticles  =   0
        self.fix_rigid_symbiont_flag        = 0

        if self.fix_rigid_symbiont:
            self.fix_rigid_symbiont_nparticles  = len(self.fix_rigid_symbiont_coordinates)
            self.fix_rigid_symbiont_interaction = fix_rigid_symbiont_interaction
            self.fix_rigid_symbiont_params      = fix_rigid_symbiont_params
            self.fix_rigid_symbiont_flag        = 1

            if len(self.fix_rigid_symbiont_params)<1:
                print("ERROR: Not enough parameters for fix rigid symbiont")
                sys.exit(1)

        ########################################################################
        #          LAMMPS SETUP AND COMMANDS REQUIRED DURING INIT              #
        #  Please do not remove the "atom_modify sort 0 0.0" from the          #  
        #  LAMMPS input file. This is required for TriMEM                      #
        ########################################################################

        ########################################################################
        # create internal lammps instance                                      #
        # MMB note: the '-screen', 'none' suppresses                           #
        #           LAMMPS output (e.g., repeated bond information)            #
        #           It can be activated by setting debug_mode = False          #
        #           By default, debug_mode = True                              #
        #           The printing of other Trimem information is unaffected.    #
        #           If you are interested in keeping track of                  #
        #           specific quantities to keep an eye on them, you            #
        #           will have to print them explicitly.                        #
        #           Keep also in mind that since the above option removes all  #
        #           output, you may have trouble debugging simulations.        #     
        #           For that reason, it may be better to remove the commands   #
        #           while you are developping, and turn it on afterwards       #
        #           and turn it off afterwards.                                #
        #           '-log', 'none'  suppresses output to log file              #
        ########################################################################
        cmdargs=['-sf','omp']

        if not self.debug_mode:
            cmdargs.append('-screen')
            cmdargs.append('none')
            cmdargs.append('-log')
            cmdargs.append('none')

        # create LAMMPS object and python wrapper (not needed?)
        self.lmp             = lammps(cmdargs=cmdargs)
        self.L               = PyLammps(ptr=self.lmp,verbose=False)
        total_particle_types = num_particle_types

        # define atom_style
        atom_style_text = "hybrid bond charge"
        bond_dihedral_text = f"bond/types {n_bond_types}"
        # modifications to atom_style if s-layer/elastic membrane exists
        if self.slayer:
            atom_style_text = "full" # molecular (we need dihedrals) + charge
            bond_dihedral_text+=" dihedral/types 1"
            bond_dihedral_text+=" extra/dihedral/per/atom 14"
        
        # pass boundary conditions - periodic bc are useless (triangulated membrane does not do well across boundary)
        boundary_string = self.parse_boundary(periodic)

        # basic system initialization
        basic_system = dedent(f"""\
            units lj
            dimension 3
            package omp 0
            boundary {boundary_string}

            atom_style    {atom_style_text}
            atom_modify sort 0 0.0

            region box block {self.algo_params.box[0]} {self.algo_params.box[1]} {self.algo_params.box[2]} {self.algo_params.box[3]} {self.algo_params.box[4]} {self.algo_params.box[5]}
            create_box {total_particle_types} box {bond_dihedral_text} extra/bond/per/atom 100 extra/special/per/atom 100

            run_style verlet

            # TriMEM computation of the forces
            fix ext all external pf/callback 1 1

            timestep {self.algo_params.step_size}

            {block(bond_text)}

            dielectric  1.0
            compute th_ke all ke
            compute th_pe all pe pair bond
            
            thermo {self.algo_params.traj_steps}
            thermo_style custom c_th_pe c_th_ke
            thermo_modify norm no

            #info styles compute out log
            echo screen

        """)

        # introduce preamble/basic system description in lammps
        self.lmp.commands_string(basic_system)

        # LAMMPS ...............................................................
        # INPUT FILE DEFINITION
        # LAMMPS ...............................................................

        # extract bond topology from Trimesh object (Python library)
        self.edges   = trimesh.Trimesh(vertices=self.mesh.x, faces=self.mesh.f).edges_unique
        self.edges   = np.unique(self.edges,axis=0)
        self.n_edges = self.edges.shape[0]

        # create (re)initialization file
        with open('sim_setup.in', 'w') as f:

            f.write('\n\n')
            f.write(f'{self.mesh.x.shape[0]+self.beads.n_beads+self.n_slayer+self.fix_rigid_symbiont_nparticles} atoms\n')
            f.write(f'{num_particle_types} atom types\n')
            f.write(f'{self.edges.shape[0]+n_tethers+self.n_slayer_bonds} bonds\n')
            f.write(f'{n_bond_types} bond types\n\n')

            # include dihedrals for s-layer
            if self.slayer:
                f.write(f'{self.n_slayer_dihedrals} dihedrals\n')
                f.write(f'1 dihedral types\n\n')

            # particle masses
            f.write('Masses\n\n')
            for part_types in range(num_particle_types):
                f.write(f'{part_types+1} {self.mass_particle_type[part_types]}\n')
            f.write('\n')

            # [COORDINATES] STANDARD SIMULATION SET-UP: only membrane and potentially nanoparticles
            if (not self.slayer) and (not self.fix_rigid_symbiont) and (not heterogeneous_membrane):
                f.write(f'Atoms # hybrid\n\n')
                for i in range(self.n_vertices):
                    f.write(f'{i + 1} 1  {self.mesh.x[i, 0]} {self.mesh.x[i, 1]} {self.mesh.x[i, 2]} 1 1.0 \n')

                # NOTE: Beads do not belong to the 'vertices' molecule and they don't have charge
                if self.beads.n_beads:
                    if self.beads.n_bead_types>1:
                        for i in range(self.beads.n_beads):
                            f.write(f'{self.n_vertices+1+i} {self.beads.types[i]} {self.beads.positions[i,0]} {self.beads.positions[i,1]} {self.beads.positions[i,2]} 0 0\n')
                    else:
                        for i in range(self.beads.n_beads):
                            f.write(f'{self.n_vertices+1+i} 2 {self.beads.positions[i,0]} {self.beads.positions[i,1]} {self.beads.positions[i,2]} 0 0\n')

            # [COORDINATES] EXTENSION: SIMULATION WITH AN SLAYER
            elif self.slayer:
                f.write(f'Atoms # full\n\n')
                # membrane vertices (type 1)
                for i in range(self.n_vertices):
                    f.write(f'{i + 1} 0 1 1.0 {self.mesh.x[i, 0]} {self.mesh.x[i, 1]} {self.mesh.x[i, 2]} \n')
                # slayer vertices (type 2)
                for i in range(self.n_slayer):
                    f.write(f'{self.n_vertices+i+1} 0 2 1.0 {self.slayer_points[i, 0]} {self.slayer_points[i, 1]} {self.slayer_points[i, 2]}\n')

            # [COORDINATES] EXTENSION: SIMULATION WITH RIGID SYMBIONTS
            elif self.fix_rigid_symbiont:

                f.write(f'Atoms # hybrid\n\n')
                # membrane particles are type 1
                for i in range(self.n_vertices):
                    f.write(f'{i + 1} 1  {self.mesh.x[i, 0]} {self.mesh.x[i, 1]} {self.mesh.x[i, 2]} 1 1.0 \n')
                # symbiont/nanoparticles particles are type 2
                for i in range(self.fix_rigid_symbiont_nparticles):
                    f.write(f'{self.n_vertices + i + 1} 2  {self.fix_rigid_symbiont_coordinates[i, 0]} {self.fix_rigid_symbiont_coordinates[i, 1]} {self.fix_rigid_symbiont_coordinates[i, 2]} 1 1.0 \n')

            # [COORDINATES] EXTENSION: SIMULATION WITH HETEROGENEOUS (2 TYPE) MEMBRANE
            elif heterogeneous_membrane:
                f.write(f'Atoms # hybrid\n\n')
                for i in range(self.n_vertices):
                    if i in heterogeneous_membrane_id:
                        f.write(f'{i + 1} 2 {self.mesh.x[i, 0]} {self.mesh.x[i, 1]} {self.mesh.x[i, 2]} 1 1.0 \n')
                    else:
                        f.write(f'{i + 1} 1 {self.mesh.x[i, 0]} {self.mesh.x[i, 1]} {self.mesh.x[i, 2]} 1 1.0 \n')

                if self.beads.n_beads:
                    if self.beads.n_bead_types>1:
                        for i in range(self.beads.n_beads):
                            f.write(f'{self.n_vertices+1+i} {self.beads.types[i]} {self.beads.positions[i,0]} {self.beads.positions[i,1]} {self.beads.positions[i,2]} 0 1.0\n')
                    else:
                        for i in range(self.beads.n_beads):
                            f.write(f'{self.n_vertices+1+i} 3 {self.beads.positions[i,0]} {self.beads.positions[i,1]} {self.beads.positions[i,2]} 0 1.0\n')

            # [VELOCITIES]
            f.write(f'Velocities \n\n')
            # membrane particles are type 1
            for i in range(self.n_vertices):
                f.write(f'{i + 1} 0.0 0.0 0.0 \n')

            # bead velocities
            if self.beads.n_beads:
                if np.any(self.beads.velocities):
                    for i in range(self.beads.n_beads):
                        f.write(f'{self.n_vertices+1+i} {self.beads.velocities[i, 0]} {self.beads.velocities[i, 1]} {self.beads.velocities[i, 2]}\n')
                else:
                    for i in range(self.beads.n_beads):
                        f.write(f'{self.n_vertices+1+i} {0} {0} {0}\n')
            # [BONDS]
            if self.slayer == False:
                f.write(f'Bonds # zero special\n\n')
            else:
                f.write(f'Bonds # hybrid\n\n')

            # [BONDS] first type of bond -- for the fluid membrane
            for i in range(self.edges.shape[0]):
                f.write(f'{i + 1} 1 {self.edges[i, 0] + 1} {self.edges[i, 1] + 1}\n')
            # [BONDS] second type of bond -- for the slayer
            for i in range(self.n_slayer_bonds):
                f.write(f'{self.edges.shape[0] + i + 1} 2 {self.slayer_bonds[i, 0] + self.n_vertices} {self.slayer_bonds[i, 1] + self.n_vertices}\n')

            if add_tether:
                for i in range(n_tethers):
                    d_temp=10^6
                    h=0
                    for j in range(self.n_vertices):
                        d_temp2=np.sum((self.mesh.x[j,:]-self.beads.positions[0,:])**2)
                        if d_temp>d_temp2:
                            d_temp=d_temp2
                            h=j

                    f.write(f'{self.edges.shape[0]+1+i} 2 {h+1} {self.n_vertices+1+i}\n')

            # [DIHEDRALS] Only applicable to S-LAYER/elastic membrane
            if self.slayer:
                f.write(f'Dihedrals # harmonic\n\n')
                for i in range(self.n_slayer_dihedrals):
                    f.write(f'{i + 1} 1 {self.slayer_dihedrals[i, 0] +self.n_vertices} {self.slayer_dihedrals[i, 1]  +self.n_vertices} {self.slayer_dihedrals[i, 2]  +self.n_vertices} {self.slayer_dihedrals[i, 3]  +self.n_vertices}\n')

        # pass all the initial configuration data to LAMMPS to read
        self.lmp.command('read_data sim_setup.in add merge')

        # LAMMPS ...............................................................
        # TABLE FOR SURFACE REPULSION
        # Create python file implementing the repulsive part of the tether potential
        # as surface repulsion readable by the lammps pair_style
        # python uses the pair_style defined above to create a lookup
        # table used as actual pair_style in the vertex-vertex interaction in Lammps
        # is subject to 1-2, 1-3 or 1-4 neighbourhood exclusion of special bonds
        # used to model the mesh topology
        # LAMMPS ...............................................................

        with open('trilmp_srp_pot.py','w') as f:

            f.write(dedent(f"""\
            import numpy as np

            class LAMMPSPairPotential(object):
                def __init__(self):
                    self.pmap=dict()
                    self.units='lj'
                def map_coeff(self,name,ltype):
                    self.pmap[ltype]=name
                def check_units(self,units):
                    if (units != self.units):
                        raise Exception("Conflicting units: %s vs. %s" % (self.units,units))

            class SRPTrimem(LAMMPSPairPotential):
                def __init__(self):
                    super(SRPTrimem,self).__init__()
                    # set coeffs: kappa_r, cutoff, r (power)
                    #              4*eps*sig**12,  4*eps*sig**6
                    self.units = 'lj'
                    self.coeff = {{'C'  : {{'C'  : ({self.eparams.repulse_params.lc1},{self.eparams.kappa_r},{self.eparams.repulse_params.r})  }} }}

                def compute_energy(self, rsq, itype, jtype):
                    coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]

                    srp1 = coeff[0]
                    srp2 = coeff[1]
                    srp3 = coeff[2]
                    r = np.sqrt(rsq)
                    rl=r-srp1

                    e=0.0
                    e+=np.exp(r/rl)
                    e/=r**srp3
                    e*=srp2

                    return e

                def compute_force(self, rsq, itype, jtype):
                    coeff = self.coeff[self.pmap[itype]][self.pmap[jtype]]
                    srp1 = coeff[0]
                    srp2 = coeff[1]
                    srp3 = coeff[2]

                    r = np.sqrt(rsq)
                    f=0.0

                    rp = r ** (srp3 + 1)
                    rl=r-srp1
                    f=srp1/(rl*rl)+srp3/r
                    f/=rp
                    f*=np.exp(r/rl)
                    f*=srp2

                    return f
            """))

        # [INTERACTION POTENTIAL] - SINGLE PARTICLE IN MEMBRANE
        if not heterogeneous_membrane:
            # write down the table for surface repulsion
            self.lmp.commands_string(dedent(f"""\
            pair_style python {self.eparams.repulse_params.lc1}
            pair_coeff * * trilmp_srp_pot.SRPTrimem {'C '*self.num_particle_types}
            shell rm -f trimem_srp.table
            pair_write  1 1 2000 rsq 0.000001 {self.eparams.repulse_params.lc1} trimem_srp.table trimem_srp 1.0 1.0
            pair_style none
            """))

            # get the 'table' interaction to work
            self.lmp.commands_string(dedent(f"""\
            pair_style hybrid/overlay table linear 2000 lj/cut 2.5
            pair_modify pair table special lj/coul 0.0 0.0 0.0 tail no
            pair_coeff 1 1 table trimem_srp.table trimem_srp
            pair_coeff * * lj/cut 0 0 0
            """))
        
        # [EXTENSION] [INTERACTION POTENTIAL] - TWO PARTICLES IN MEMBRANE
        if heterogeneous_membrane:
            # write down the table for surface repulsion
            self.lmp.commands_string(dedent(f"""\
            pair_style python {self.eparams.repulse_params.lc1}
            pair_coeff * * trilmp_srp_pot.SRPTrimem {'C '*self.num_particle_types}
            shell rm -f trimem_srp.table
            pair_write  1 1 2000 rsq 0.000001 {self.eparams.repulse_params.lc1} trimem_srp.table trimem_srp11 1.0 1.0
            pair_write  2 2 2000 rsq 0.000001 {self.eparams.repulse_params.lc1} trimem_srp.table trimem_srp22 1.0 1.0
            pair_write  1 2 2000 rsq 0.000001 {self.eparams.repulse_params.lc1} trimem_srp.table trimem_srp12 1.0 1.0
            pair_style none
            """))

            # get the 'table' interaction to work
            self.lmp.commands_string(dedent(f"""\
            pair_style hybrid/overlay table linear 2000 lj/cut 2.5
            pair_modify pair table special lj/coul 0.0 0.0 0.0 tail no
            pair_coeff 1 1 table trimem_srp.table trimem_srp11
            pair_coeff 2 2 table trimem_srp.table trimem_srp22
            pair_coeff 1 2 table trimem_srp.table trimem_srp12
            pair_coeff * * lj/cut 0 0 0
            """))

        # LAMMPS ...............................................................
        # GROUPS
        # LAMMPS ...............................................................

        # define particle groups
        for pg in range(num_particle_types):
            self.lmp.command(f'group {group_particle_type[pg]} type {pg+1}')

        # LAMMPS ...............................................................
        # VELOCITIES
        # LAMMPS ...............................................................
        
        
        # velocity settings
        self.atom_props = f"""
                        velocity {group_particle_type[0]} create {self.algo_params.initial_temperature} 1298371 mom yes dist gaussian
                        """
        """
        # MMB NOTE: Commenting out because we are passing velocities
        # in the sim_setup file
        # initialize random velocities if thermal velocities is chosen or set to 0
        if self.algo_params.thermal_velocities:
            self.lmp.commands_string(self.atom_props)
        else:
            self.lmp.command('velocity all zero linear')
        """

        # setting or reinitializing mesh velocities (important for pickle)
        if np.any(self.mesh_velocity):
            velocities = self.lmp.numpy.extract_atom("v")
            for i in range(self.n_vertices):
                velocities[i, :]=self.mesh_velocity[i,:]

        # LAMMPS ...............................................................
        # INTERACTIONS AND TRIMEM CALLBACK
        # LAMMPS ...............................................................

        # set callback for helfrich gradient to be handed from TRIMEM to LAMMPS via fix external "ext"
        self.lmp.set_fix_external_callback("ext", self.callback_one, self.lmp)

        # Temperature in LAMMPS set to fixed initial temperature
        self.T = self.algo_params.initial_temperature

        # initialization of system energy components for HMC (irrelevant for pureMD)
        v       = self.lmp.numpy.extract_atom("v")
        self.pe = 0.0
        self.ke = 0.5 * self.algo_params.momentum_variance*v.ravel().dot(v.ravel())
        self.he = self.estore.energy(self.mesh.trimesh)#+0.5 * v.ravel().dot(v.ravel())
        self.energy_new = 0.0

        ########################################################################
        #                           BOOK-KEEPING                               #
        ########################################################################

        # flip stats
        self.f_i = 0
        self.f_acc = 0
        self.f_num = 0
        self.f_att = 0

        # move stats
        self.m_i = 0
        self.m_acc = 0

        self.counter = Counter(move=move_count, flip=flip_count)
        self.timer   = Timer(ptime, ptimestamp, timearray, timearray_new,dtimestamp,stime)

        self.cpt_writer = self.make_checkpoint_handle()
        self.process    = psutil.Process()
        self.n          = self.algo_params.num_steps // self.output_params.info if self.output_params.info!=0 else 0.0

        self.info_step    = max(self.output_params.info, 0)
        self.out_step     = max(self.output_params.thin, 0)
        self.cpt_step     = max(self.output_params.checkpoint_every, 0)
        self.refresh_step = max(self.algo_params.refresh, 0)

        ########################################################################
        #          TRAJECTORY WRITERS (NOT USED ATM)                           #
        # Might be better to use LAMMPS' capabilities to dump desired system   #
        # properties. It can allow better handling of memory usage             #
        ########################################################################

        if self.output_params.output_format=='xyz' or self.output_params.output_format=='vtu' or self.output_params.output_format=='xdmf':
             self.output = lambda i : make_output(self.output_params.output_format, self.output_params.output_prefix,
                                  self.output_params.output_counter, callback=self.update_output_counter)
        if self.output_params.output_format == 'lammps_txt':
            def lammps_output(i):
               self.L.command(f'write_data {self.output_params.output_prefix}.s{i}.txt')
            self.output = lambda i: lammps_output(i)

        if self.output_params.output_format == 'lammps_txt_folder':
            def lammps_output(i):
                self.L.command(f'write_data lmp_trj/{self.output_params.output_prefix}.s{i}.txt')
            os.system('mkdir -p lmp_trj')
            self.output=lambda i: lammps_output(i)

        if self.output_params.output_format == 'h5_custom':
            self.h5writer=trimem.mc.trilmp_h5.H5TrajectoryWriter(self.output_params)
            self.h5writer._init_struct(self.lmp,self.mesh,self.beads,self.estore)
            self.output = lambda i: self.h5writer._write_state(self.lmp,self.mesh,i)

        ########################################################################
        #                   EDGE-FLIPPING (MEMBRANE FLUIDITY)                  #
        # MW: In this section the serial or parallel flipping method is chosen #
        # and functions forwarding the updated topology to LAMMPS are defined. #
        # We use the flip function flip_nsr/pflip_nsr which are reliant on     #
        # the the use of the estore_nsr. Hence we shut off the calculation     #
        # of surface repusion in TRIMEM. If for some reason this functionality #
        # should be needed one would have to remove all '_nsr' suffixes        #
        # and use the estore.eparams.repulse_params which are kept for         #
        # backwards portability                                                #
        ########################################################################

        # chosing function to be used for flipping
        if self.algo_params.flip_type == "none" or self.algo_params.flip_ratio == 0.0:
            self._flips = lambda: 0
        elif self.algo_params.flip_type == "serial":
            self._flips = lambda: m.flip_nsr(self.mesh.trimesh, self.estore, self.algo_params.flip_ratio)
        elif self.algo_params.flip_type == "parallel":
            self._flips = lambda: m.pflip_nsr(self.mesh.trimesh, self.estore, self.algo_params.flip_ratio)
        else:
            raise ValueError("Wrong flip-type: {}".format(self.algo_params.flip_type))
        
        print("End of initialization (or reloading).")
        ########################################################################
        #                       END OF INIT                                    #
        ########################################################################


    ############################################################################
    #               *SELF FUNCTIONS*: FLIP STEP FUNCTIONS                      #
    ############################################################################

    # test function to perform a single flip in lammps from (i,j)-> (k,l)
    def lmp_flip_single(self,i,j,k,l):

        self.lmp.command(f'group flip_off id {i} {j}')
        self.lmp.command('delete_bonds flip_off bond 1 remove')
        self.lmp.command('group flip_off clear')
        self.lmp.command(f'create_bonds single/bond 1 {k} {l}')

    # function used for flipping
    def lmp_flip(self,flip_id):


        nf=flip_id[-1][0]
        
        """
        # MMB CHANGED
        if np.any(self.vertices_at_edge):
            # Check for elements in B
            mask = np.isin(flip_id, self.vertices_at_edge)
            print(mask)
            # Check rows that do not contain vertex at the edge
            rows_without_edge_vertices = ~mask.any(axis=1)
            print(rows_without_edge_vertices)
            indices = np.where(rows_without_edge_vertices)[0]
            if len(indices)>0:
                # Filter rows
                flip_id = np.array(flip_id)
                flip_id = flip_id[indices]
                nf = len(flip_id)
            else:
                nf = 0
        """
        
        if nf:

            del_com='remove'
            
            # 'test_mode' allows for time optimization (less neighbour lists built)
            if self.test_mode:

                for i in range(nf-1):
                    if i == nf-1:
                        del_com = 'remove special'

                    # ---------------------------------------------------
                    # ABOUT THESE COMMANDS (see LAMMPS documentation)
                    # - delete_bonds
                    #   'remove' keyword -> 'adjusts the global bond count'
                    #   'special' keyword -> re-computes pairwise weighting list
                    #                        weighting list treats turned-off bonds the same as turned-on
                    #                        turned-off bonds have to be removed to change the weighting list

                    # REGULAR BONDS: DO NOT trigger internal list creation now (create_bonds)
                    self.lmp.command(f'create_bonds single/bond 1 {flip_id[i][0] + 1} {flip_id[i][1] + 1} special no') # the 'special no' here prevents the weighting list from being computed
                    self.lmp.command(f'group flip_off id {flip_id[i][2] + 1} {flip_id[i][3] + 1}') # you MUST create a group on which delete_bonds will adct
                    self.lmp.command(f'delete_bonds flip_off bond 1 remove') # you must (?) remove the bonds you are turning off
                    self.lmp.command('group flip_off clear') # you must clear the group or else particles will be added to it
                
                # LAST BOND: TRIGGER INTERNAL LIST CREATION NOW (create_bonds)
                self.lmp.command(f'create_bonds single/bond 1 {flip_id[nf-1][0] + 1} {flip_id[nf-1][1] + 1} special yes') # the 'special yes' here is invoked to compute the weighting list
                self.lmp.command(f'group flip_off id {flip_id[nf-1][2] + 1} {flip_id[nf-1][3] + 1}')
                self.lmp.command(f'delete_bonds flip_off bond 1 remove special') # maybe the special is not needed here?
                self.lmp.command('group flip_off clear')

            # original implementation - more time-consuming
            else:

                for i in range(nf):
                    if i == nf-1:
                        del_com = 'remove special'

                    self.lmp.command(f'create_bonds single/bond 1 {flip_id[i][0] + 1} {flip_id[i][1] + 1}')
                    self.lmp.command(f'group flip_off id {flip_id[i][2] + 1} {flip_id[i][3] + 1}')
                    self.lmp.command(f'delete_bonds flip_off bond 1 {del_com}')
                    self.lmp.command('group flip_off clear')

        else:
            pass

    # print flip information
    def flip_info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())
        if self.output_params.info and i_total % self.output_params.info == 0:

            # MMB CHANGED -- MAKING SURE THAT NUMBER OF EDGES KNOWS ABOUT THE FLIP RATIO
            n_edges = self.mesh.trimesh.n_edges()*self.algo_params.flip_ratio
            ar      = self.f_acc / (self.f_i * n_edges) if not self.f_i == 0 else 0.0
            self.acceptance_rate = ar

            print(f"Number of candidate edges: {n_edges}")
            print(f"Accepted: {self.f_acc}")
            print(f"\n-- MCFlips-Step {self.counter['flip']}")
            print(f"----- flip-accept: {ar}")
            print(f"----- flip-rate:   {self.algo_params.flip_ratio}")

            self.f_acc = 0
            self.f_i   = 0
            self.f_num = 0
            self.f_att = 0

    # the actual flip step
    def flip_step(self):
        
        # only do a flip step if the ratio is non-zero
        if self.algo_params.flip_ratio!=0:

            if self.debug_mode:
                # time the flipping
                start = time.time()

            # make one step
            flip_ids=self._flips()

            if self.debug_mode:
                time_flips_trimem = time.time()
            
            self.mesh.f[:]=self.mesh.f
            self.lmp_flip(flip_ids)

            if self.debug_mode:
                time_flips_lammps = time.time()

            self.f_acc += flip_ids[-1][0]
            self.f_num += flip_ids[-1][1]
            self.f_att += flip_ids[-1][2]
            self.f_i += 1
            self.counter["flip"] += 1

            if self.debug_mode:
                end = time.time()
                ff = open("TriLMP_optimization.dat", "a+")
                ff.writelines(f"MC {end-start} {time_flips_trimem - start} {time_flips_lammps - start} {len(flip_ids)}\n")
                ff.close()

    ############################################################################
    #             *SELF FUNCTIONS*: HYBRID MONTE CARLO (MC + MD) SECTION       #
    # MW: In this section we combine all functions that are used for           #
    # either HMC/pure_MD run or minimize including the wrapper                 #
    # functions used for updating the mesh (on the TRIMEM side)                #
    # when evaluating the gradient                                             #
    ############################################################################

    # MD stage of the code
    def hmc_step(self):

        # in practice never used - MMB UNTOUCHED
        if not self.algo_params.pure_MD:

            # setting temperature
            i = sum(self.counter.values())
            Tn = np.exp(-self.algo_params.cooling_factor * (i - self.algo_params.start_cooling)) * self.algo_params.initial_temperature
            self.T = max(min(Tn, self.algo_params.initial_temperature), 10e-4)

            # safe mesh for reset in case of rejection
            self.mesh_temp=copy(self.mesh.x)
            self.beads_temp=copy(self.beads.positions)

            #calute energy - future make a flag system to avoid double calculation if lst step was also hmc step
            if self.algo_params.thermal_velocities:
                 self.atom_props = f"""velocity        vertices create {self.T} {np.random.randint(1,9999999)} mom yes dist gaussian"""
                 self.lmp.commands_string(self.atom_props)
                 v = self.lmp.numpy.extract_atom("v")
                 self.ke= 0.5 * (self.masses[:,np.newaxis]*v).ravel().dot(v.ravel())
            else:
                self.velocities_temp=self.lmp.numpy.extract_atom('v')

            # use ke from lammps to get kinetic energy
            self.he = self.estore.energy(self.mesh.trimesh)
            self.energy = self.pe + self.ke + self.he

            #run MD trajectory
            #self.lmp.command(f'run {self.algo_params.traj_steps} post no')
            self.L.run(self.algo_params.traj_steps)

            #set global energy in lammps
            #self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))

            # calculate energy difference -> future: do it all in lamps via the set command above (incl. SP and bead interactions)
            if not self.beads.n_beads:
                self.mesh.x[:] = self.lmp.numpy.extract_atom("x")
            else:
                pos_alloc=self.lmp.numpy.extract_atom("x")
                self.mesh.x[:] = pos_alloc[:self.n_vertices]
                self.beads.positions[:] = pos_alloc[self.n_vertices:self.n_vertices+self.beads.n_beads]

            # kinetic and potential energy via LAMMPS
            self.ke_new=self.lmp.numpy.extract_compute("th_ke",LMP_STYLE_GLOBAL,LMP_TYPE_SCALAR)
            self.pe_new=self.lmp.numpy.extract_compute("th_pe", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)

            # add helfrich energy via Trimem
            self.energy_new = self.estore.energy(self.mesh.trimesh) + self.ke_new + self.pe_new

            dh = (self.energy_new- self.energy) / self.T
            print(dh)

            # compute acceptance probability: min(1, np.exp(-de))
            a = 1.0 if dh <= 0 else np.exp(-dh)
            u = np.random.uniform()
            acc = u <= a
            if acc:
                self.m_acc += 1
                self.ke=copy(self.ke_new)
                self.pe=copy(self.pe_new)

            else:
                # reset positions if rejected
                if not self.beads.n_beads:
                    self.mesh.x[:]=self.mesh_temp[:]
                    atoms_alloc=self.L.atoms
                    if self.algo_params.thermal_velocities:
                        for i in range(self.n_vertices):
                            atoms_alloc[i].position[:]=self.mesh_temp[i,:]
                    else:
                        for i in range(self.n_vertices):
                            atoms_alloc[i].position[:]=self.mesh_temp[i,:]
                            atoms_alloc[i].velocity[:]=self.velocities_temp[i,:]
                else:

                    self.mesh.x[:] = self.mesh_temp[:]
                    self.beads.positions[:] = self.beads_temp[:]
                    atoms_alloc = self.L.atoms

                    if self.algo_params.thermal_velocities:
                        for i in range(self.n_vertices):
                            atoms_alloc[i].position[:] = self.mesh_temp[i, :]

                        for i in range(self.n_vertices,self.n_vertices+self.beads.n_beads):
                            atoms_alloc[i].position[:] = self.beads_temp[i-self.n_vertices, :]
                    else:
                        for i in range(self.n_vertices):
                            atoms_alloc[i].position[:] = self.mesh_temp[i, :]
                            atoms_alloc[i].velocity[:] = self.velocities_temp[i,:]

                        for i in range(self.n_vertices, self.n_vertices + self.beads.n_beads):
                            atoms_alloc[i].position[:] = self.beads_temp[i - self.n_vertices, :]
                            atoms_alloc[i].velocity[:] = self.velocities_temp[i, :]

            # CAVEAT using thermal_velocities=False and pure_MD=False can result in a deadlock
            # UPDATE COUNTERS
            self.m_i += 1
            self.counter["move"] += 1

        # pure MD run of the code
        else:
            
            if self.debug_mode:
                # MMB timing purposes
                start = time.time()

            self.lmp.command(f'run {self.algo_params.traj_steps}')

            self.m_acc += 1
            self.m_i += 1
            self.counter["move"] += 1

            if self.debug_mode:

                # MMB timing purposes
                end = time.time()
                ff = open("TriLMP_optimization.dat", "a+")
                ff.writelines(f"MD {end-start} {-1} {-1} {-1}\n")
                ff.close()

    # print MD stage information
    def hmc_info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())

        if self.output_params.info and i_total % self.output_params.info == 0:
            ar = self.m_acc / self.m_i if not self.m_i == 0 else 0.0
            print("\n-- HMC-Step ", self.counter["move"])
            print("----- acc-rate:   ", ar)
            print("----- temperature:", self.T)
            self.m_acc = 0
            self.m_i = 0

    ############################################################################
    #                        *SELF FUNCTIONS*: RUN                             #                                            #
    ############################################################################

    def halt_symbiont_simulation(self, step, check_outofrange, check_outofrange_freq, check_outofrange_cutoff):
        if (self.equilibrated) and (check_outofrange) and (step%check_outofrange_freq ==0):
            pos_alloc=self.lmp.numpy.extract_atom("x")
            self.mesh.x[:] = np.array(pos_alloc[:self.n_vertices])
            self.beads.positions[:] = np.array(pos_alloc[self.n_vertices:self.n_vertices+self.beads.n_beads])
            distance_beads_nanoparticle = np.sqrt((self.mesh.x[:, 0] - self.beads.positions[0,0])**2 + (self.mesh.x[:, 1] - self.beads.positions[0,1])**2 + (self.mesh.x[:, 2] - self.beads.positions[0,2])**2)
            if np.all(distance_beads_nanoparticle > check_outofrange_cutoff):
                fileERROR = open("ERROR_REPORT.dat", "w")
                fileERROR.writelines(f"Nanoparticle reached cutoff range at step {step}")
                fileERROR.close()
                sys.exit(0)

    def step_random(self):

        """ Make one step each with each algorithm """
        if np.random.choice(2) == 0:
            t_fix = time.time()
            self.hmc_step()
            self.timer.timearray_new[0] += (time.time() - t_fix)

            # after performing the MD section of the simulation, update counter
            self.MDsteps +=self.traj_steps

        else:
            t_fix = time.time()
            self.flip_step()
            self.timer.timearray_new[1] += (time.time() - t_fix)

    def step_alternate(self):

        """
        Make one step each with each algorithm.
        (note that in general this functionality is not used)
        """

        t_fix = time.time()
        self.hmc_step()
        # after performing the MD section of the simulation, update counter
        self.MDsteps +=self.traj_steps
        self.timer.timearray_new[0] += (time.time() - t_fix)
        t_fix = time.time()
        
        self.flip_step()
        self.timer.timearray_new[1] += (time.time() - t_fix)

    def raise_errors_run(self, integrators_defined, check_outofrange, check_outofrange_freq, check_outofrange_cutoff, fix_symbionts_near, evaluate_inside_membrane, factor_inside_membrane, naive_compression, desired_interlayer_distance, ghost_membrane_consumes):

        """
        Use function to raise errors and prevent compatibility issues
        """
        
        if check_outofrange and (check_outofrange_freq<0 or check_outofrange_cutoff<0):
            print("TRILMP ERROR: Incorrect check_outofrange parameters")
            sys.exit(1)

        if self.slayer and self.n_beads!=0:
            print("TRILMP ERROR: Slayer is true, but n_beads is non zero. This simulation set-up is not possible.")
            sys.exit(1)

        if fix_symbionts_near and self.beads.n_bead_types==0:
            print("TRILMP ERROR: No symbiont to place near.")
            sys.exit(1)

        if not integrators_defined:
            print("TRILMP ERROR: You have not defined a single integrator.")
            sys.exit(1)

        if (self.equilibration_rounds>0) and (self.equilibration_rounds%self.traj_steps!=0):
            print("TRILMP ERROR: Number of equilibration rounds not a multiple of traj_steps. Post equilibration commands will never be read.")
            sys.exit(1)

        if(evaluate_inside_membrane) and (factor_inside_membrane == 0):
            print("TRILMP ERROR: You want to evaluate inside the membrane but you have provided a factor of 0.")
            sys.exit(1)

        if(not naive_compression) and (desired_interlayer_distance == 0):
            print("TRILMP ERROR: Desired interlayer distance for complex compression cannot be zero.")
            sys.exit(1)

        if(ghost_membrane_consumes) and (not evaluate_inside_membrane):
            print("TRILMP ERROR: You want the ghost membrane to consume but you are not setting up the evaluation inside the membrane.")
            sys.exit(1)

        print("No errors to report for run. Simulation begins.")

    def run(
            self, N=0, integrators_defined = False, check_outofrange = False, 
            check_outofrange_freq = -1, check_outofrange_cutoff = -1, fix_symbionts_near = True, 
            postequilibration_lammps_commands = None, seed = 123, current_step = 0,
            step_dependent_protocol = False, step_protocol_commands = None, step_protocol_frequency = 0,
            steps_in_protocol = 0, evaluate_configuration = False, evaluate_inside_membrane = False,
            factor_inside_membrane=0, naive_compression = True, desired_interlayer_distance=0, 
            gcmc_by_hand=False, desired_particles_source=0, pure_sink=False, desired_particles_sink=0,
            ghost_membrane_consumes = False, cutoff_consumption = 0, move_membrane=True, force_field_normals = False,
            A_force =0 , B_force =0,
        ):

        """
        MAIN TRILMP FUNCTION: Combine MD + MC runs

        Parameters:

        - N : number of program steps
        

        - check_outofrange : check whether single symbiont is too far from membrane
            Additional parameters:
                - check_outofrange_freq: how often to check for event
                - check_outofrange_cutoff: when to consider (distance-wise) event has happened

        - fix_symbionts_near : place symbionts within interaction range of membrane
        """

        self.move_membrane = move_membrane
        self.force_field_normals = force_field_normals
        self.A_force = A_force
        self.B_force = B_force

        # set the numpy seed (MD + MC stepping)
        np.random.seed(seed=seed)

        # -------------------------------------------------
        # clear up the file for the timing
        if self.debug_mode:
            fTiming = open("TriLMP_optimization.dat", "w")
            fTiming.writelines(f"MOVE TOTAL FLIP_TRIMEM FLIP_LAMMPS NUM_FLIPS\n")
            fTiming.close()

        # clear up this file
        temp_file = open(f'{self.output_params.output_prefix}_system.dat','w')
        temp_file.close()

        # clear up this file
        temp_file = open(f'{self.output_params.output_prefix}_performance.dat','w')
        temp_file.close()
        # -------------------------------------------------
        
        # check whether there is any initialization error
        self.raise_errors_run(integrators_defined, check_outofrange, check_outofrange_freq, check_outofrange_cutoff, fix_symbionts_near, evaluate_inside_membrane, factor_inside_membrane, naive_compression, desired_interlayer_distance, ghost_membrane_consumes)

        # determine length simulation
        if N==0:
            N=self.algo_params.num_steps
        if self.algo_params.switch_mode=='random':
            self.step = lambda: self.step_random()
        elif self.algo_params.switch_mode=='alternating':
            self.step = lambda: self.step_alternate()
        else:
            raise ValueError("Wrong switchmode: {}. Use 'random' or 'alternating' ".format(self.algo_params.flip_type))
        
        # counters for MD steps
        i = -1
        self.MDsteps = 0
        oldsteps = -1

        if current_step:
            self.MDsteps = current_step
            N += current_step

        # initial conditions -- record
        self.callback(np.copy(self.mesh.x),self.counter)

        if step_dependent_protocol:
            applied_protocol = 0

        # we just want to evaluate a configuration and exit the run
        if evaluate_configuration:
            # do not integrate equations of motion, just evaluate
            self.lmp.command(f'run 0')
            # exit - do not continue running
            return
        
        # run simulation for dictated number
        # of MD steps
        while self.MDsteps<N:

            # counter for updates and so on
            i+=1

            # the program MC + MD
            self.step()
            self.hmc_info()
            self.callback(np.copy(self.mesh.x),self.counter)
            self.flip_info()

            # check if simulation must stop
            self.halt_symbiont_simulation(i, check_outofrange, check_outofrange_freq, check_outofrange_cutoff)

            # post equilibration update, if it applies
            if self.MDsteps==self.equilibration_rounds and self.equilibrated == False:

                if self.debug_mode:
                    print("These are your current fixes pre equilibration: ")
                    print(self.L.fixes)

                """
                # write down existing fixes
                with open('LAMMPSfixes.dat', 'w') as f:
                    f.writelines('Pre-equilibration: \n')
                    for fix in self.L.fixes:
                        f.writelines(f'{fix}\n')
                    f.writelines('\n')
                    f.close()
                """

                # place symbionts near
                if fix_symbionts_near:

                    # extract the coordinates of the membrane vertices and beads
                    pos_alloc=self.lmp.numpy.extract_atom("x")
                    self.mesh.x[:] = pos_alloc[:self.n_vertices]
                    self.beads.positions[:] = pos_alloc[self.n_vertices:self.n_vertices+self.beads.n_beads]

                    # interaction single symbiont
                    if self.beads.n_beads==1:
                        coord_bead = self.mesh.x[0]
                        rtemp      = np.sqrt(coord_bead[0]**2 + coord_bead[1]**2 + coord_bead[2]**2)
                        buffering  = 1.1
                        sigma_tilde = 0.5*(1+self.beads.bead_sizes[0])
                        x = coord_bead[0] + buffering*sigma_tilde*coord_bead[0]/rtemp
                        y = coord_bead[1] + buffering*sigma_tilde*coord_bead[1]/rtemp
                        z = coord_bead[2] + buffering*sigma_tilde*coord_bead[2]/rtemp

                        # make sure that lammps knows about this
                        pos_alloc[self.n_vertices, 0] = x 
                        pos_alloc[self.n_vertices, 1] = y 
                        pos_alloc[self.n_vertices, 2] = z 

                        """
                        atoms_alloc = self.L.atoms
                        atoms_alloc[self.n_vertices].position[0] = x
                        atoms_alloc[self.n_vertices].position[1] = y
                        atoms_alloc[self.n_vertices].position[2] = z
                        """
                        
                    # multiple symbionts on shell
                    elif self.beads.n_beads>1:
                        buffering =1.05
                        sigma_tilde = 0.5*(1+self.beads.bead_sizes)
                        n = self.beads.n_beads
                        goldenRatio = (1+5**0.5)/2
                        i = np.arange(0, n)
                        theta = 2*np.pi*i/goldenRatio
                        phi = np.arccos(1-2*(i+0.5)/n)
                        r = np.sqrt(self.mesh.x[0, 0]**2 + self.mesh.x[0, 1]**2 + self.mesh.x[0, 2]**2)
                        r += 1.1*sigma_tilde
                        x, y, z = r*np.cos(theta)*np.sin(phi), r*np.sin(theta)*np.sin(phi), r*np.cos(phi)

                        atoms_alloc = self.L.atoms
                        for q in range(self.beads.n_beads):
                            min_distance = 1000
                            xx, yy, zz = x[q], y[q], z[q]
                            for qq in range(self.n_vertices):
                                xv = self.mesh.x[qq, 0]
                                yv = self.mesh.x[qq, 1]
                                zv = self.mesh.x[qq, 2]
                                distance = np.sqrt((xv-xx)**2 + (yv-yy)**2 + (zv-zz)**2)
                                if distance < min_distance:
                                    min_distance = distance
                                    index = qq

                            xtemp = self.mesh.x[index, 0]
                            ytemp = self.mesh.x[index, 1]
                            ztemp = self.mesh.x[index, 2]
                            rtemp = np.sqrt(xtemp**2 + ytemp**2 + ztemp**2)

                            pos_alloc[self.n_vertices+q, 0] = xtemp + 1.05*sigma_tilde*xtemp/rtemp
                            pos_alloc[self.n_vertices+q, 1] = ytemp + 1.05*sigma_tilde*ytemp/rtemp
                            pos_alloc[self.n_vertices+q, 2] = ztemp + 1.05*sigma_tilde*ztemp/rtemp

                            """
                            atoms_alloc[self.n_vertices+q].position[0] = xtemp + 1.05*sigma_tilde*xtemp/rtemp
                            atoms_alloc[self.n_vertices+q].position[1] = ytemp + 1.05*sigma_tilde*ytemp/rtemp
                            atoms_alloc[self.n_vertices+q].position[2] = ztemp + 1.05*sigma_tilde*ztemp/rtemp
                            """
                            
                # change status of the membrane
                self.equilibrated = True

                # add commands that you would like LAMMPS to know of after the equilibration
                if postequilibration_lammps_commands:
                    for command in postequilibration_lammps_commands:
                        self.lmp.command(command)

                """
                # save post-equilibration fixes
                with open('LAMMPSfixes.dat', 'a+') as f:
                    f.writelines('POST-equilibration: \n')
                    for fix in self.L.fixes:
                        f.writelines(f'{fix}\n')
                    f.writelines('\n')
                    f.close()
                """

                if self.debug_mode:
                    print("These are your current fixes: ")
                    print(self.L.fixes)

            # protocols that rely on LAMMPS commands which have to be added after equilibration
            if (self.equilibrated) and (step_dependent_protocol):
                # if it is the frequency at which we want to apply the protocol
                if (self.MDsteps % step_protocol_frequency == 0) and self.MDsteps!= oldsteps:
                    # if we have not applied all protocol steps
                    if applied_protocol<steps_in_protocol:
                        for command in step_protocol_commands[applied_protocol]:
                            self.lmp.command(command)
                        # check what is the protocol step that has been applied
                        applied_protocol+=1 

                oldsteps = self.MDsteps

            # evaluation of inside of membrane and application of corresponding protocol
            if evaluate_inside_membrane:
                
                # extract the particle coordinates (more to pass them later on)
                pos_alloc=self.lmp.numpy.extract_atom("x")

                new_mesh = trimesh.Trimesh(vertices=self.mesh.x, faces=self.mesh.f)
                baricenters = new_mesh.triangles_center
                face_normals = new_mesh.face_normals
                compressed_membrane = baricenters - desired_interlayer_distance*face_normals

                # inform lammps of the coordinates again
                pos_alloc[self.n_vertices:self.n_vertices+self.n_faces] = compressed_membrane

                if ghost_membrane_consumes:
                    self.lmp.command(f'delete_atoms overlap {cutoff_consumption} metabolites inside')

            # do gcmc by hand rather than using fix gcmc
            if (self.MDsteps>(self.equilibration_rounds+self.traj_steps)) and gcmc_by_hand:
                
                to_add = 0
                to_delete = 0
                
                # --------------------
                # SOURCE
                # --------------------

                # count how many particles are there in the source
                particles_source = self.lmp.numpy.extract_compute("countsource", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)

                # add needed particles in source (assuming 3 particle types)
                to_add = desired_particles_source - particles_source[2]

                # if you need to add particles because there are not enough
                if to_add>0:
                    self.lmp.command(f'create_atoms 3 random {int(to_add)} {i+1} SOURCE')
                if to_add<0:
                    self.lmp.command(f'delete_atoms random count {int(to_add*(-1))} no insource SOURCE {i+3}')

                # --------------------
                # SINK
                # --------------------

                # count how many particles there are in the sink
                if not pure_sink:
                    particles_sink = self.lmp.numpy.extract_compute("countsink", LMP_STYLE_GLOBAL, LMP_TYPE_VECTOR)
               
                # if you want to regulate the number of particles in the sink
                if not pure_sink:
                    to_delete = desired_particles_sink - particles_sink[2]

                # if you need to delete particles because there are too many
                if to_delete<0:
                    self.lmp.command(f'delete_atoms random count {int(to_delete*(-1))} no insink SINK {i+2}')

                # if you need to add particles because there are too few in the sink
                if to_delete>0:
                    self.lmp.command(f'create_atoms 3 random {int(to_delete)} {i+4} SINK')

                # delete particles in sink by default
                if pure_sink:
                    self.lmp.command(f'delete_atoms region SINK')

                # reevaluate group type 3 for correct integration
                self.lmp.command(f'group metabolites type 3')

    ############################################################################
    #                    *SELF FUNCTIONS*: WRAPPER FUNCTIONS                   #
    ############################################################################

    # decorator that updates the mesh
    def _update_mesh(func):

        """
        VARIANT FOR USE WITH self.minim()
        Decorates a method with an update of the mesh vertices.
        
        The method must have signature f(self, x, args, kwargs) with
        x being the new vertex coordinates.

        Note that it is also a decorator of the callback function.
        """

        def wrap(self, x, *args, **kwargs):
            self.mesh.x = x.reshape(self.mesh.x.shape)
            return func(self, x, *args, **kwargs)
        
        wrap.__doc__  = func.__doc__
        wrap.__name__ = func.__name__

        return wrap

    # decorator that updates the mesh
    def _update_mesh_one(func):

        """
        VARIANT FOR USE WITH LAMMPS: Decorates a method with an update of the mesh vertices.

        The method must have signature f(self, lmp, ntimestep, nlocal, tag, x,f args, kwargs) with
        x being the new vertex coordinates.
        """
        
        def wrap(self,  lmp, ntimestep, nlocal, tag, x,f,  *args, **kwargs):

            self.mesh.x = x[:self.n_vertices].reshape(self.mesh.x[:self.n_vertices].shape)
            #self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))
            return func(self, lmp, ntimestep, nlocal, tag, x,f, *args, **kwargs)
        
        wrap.__doc__  = func.__doc__
        wrap.__name__ = func.__name__
        return wrap

    # for LAMMPS force update
    @_update_mesh_one
    def callback_one(self, lmp, ntimestep, nlocal, tag, x, f):
        """
        !!!!!!! This function is used as callback to TRIMEM FROM LAMMPS !!!!!
        This is where the forces on the membrane beads, computed by TriMEM
        are extracted, and LAMMPS uses them to act on the membrane beads.
        We make it be equal, not add?
        """
        #print(tag)
        #tag_clear=[x-1 for x in tag if x <= self.n_vertices]
        if self.move_membrane:
            f[:self.n_vertices]=-self.estore.gradient(self.mesh.trimesh)

        # include a force field that acts normal to the surface of the membrane
        if self.force_field_normals:
            
            new_mesh = trimesh.Trimesh(vertices=self.mesh.x, faces=self.mesh.f)
            face_normals = new_mesh.face_normals
            mean_vertex_normals = trimesh.geometry.mean_vertex_normals(len(self.mesh.x), self.mesh.f, face_normals)

            # force field in the direction of x
            magnitude_force = self.A_force*self.mesh.x[:, 0] + self.B_force
            magnitude_force_newaxis = magnitude_force[:, np.newaxis]

            forces_to_add = mean_vertex_normals*magnitude_force_newaxis
            f[:self.n_vertices] += forces_to_add

        # if needed for flat membrane, correct
        if np.any(self.vertices_at_edge):
            f[self.vertices_at_edge] = 0.0

        ## UNCOMMENT IF TRIMEM SHOULD GET THE ENERGY IN REALTIME - MMB uncommenting
        self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))

    # for minimization
    @_update_mesh
    def fun(self, x):
        """Evaluate energy.

        Updates ``self.mesh`` with ``x`` and calls ``self.estore.energy(x)``.

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in ``self.mesh``.
            args: ignored

        Keyword Args:
            kwargs: ignored

        Returns:
            float:
                Value of the Energy represented by ``self.estore``.
        """
        return self._ravel(self.estore.energy(self.mesh.trimesh))

    @_update_mesh
    def grad(self, x):
        """Evaluate gradient.

        Updates ``self.mesh`` with ``x`` and calls ``self.estore.gradient(x)``.

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in ``self.mesh``.
            args: ignored

        Keyword Args:
            kwargs: ignored

        Returns:
            ndarray[float]:
                Gradient with respect to `x` of the Energy represented by
                ``self.estore``.
        """
        return self._ravel(self.estore.gradient(self.mesh.trimesh))

    ############################################################################
    #                    *SELF FUNCTIONS*: CALLBACK DURING .run()              #
    # This functions performs statistics/output/etc. during the simulation run #
    ############################################################################

    @_update_mesh
    def callback(self,x, steps):

        """Callback.

        Allows for the injection of custom trimem functionality into generic
        sampling and minimization algorithms:

            * stdout verbosity
            * writing of output trajectories
            * writing of checkpoint files
            * update of the internal state of self.estore

        Args:
            x (ndarray[float]): (N,3) array of vertex positions with N being
                the number of vertices in self.mesh.
            steps (collections.Counter): step counter dictionary
            args: ignored

        Keyword Args:
            kwargs: ignored
        """

        # update reference properties
        self.estore.update_reference_properties()

        i = sum(steps.values()) #py3.10: steps.total()

        if self.output_params.info and (i % self.output_params.info == 0):
            print("\n-- Energy-Evaluation-Step ", i)
            self.estore.print_info(self.mesh.trimesh)

        if self.output_params.checkpoint_every and (i % self.output_params.checkpoint_every == 0):
            # make checkpoints alternating between two points
            self.cpt_writer()
            
        if (self.MDsteps % self.output_params.info == 0):
            with open(f"checkpoints/ckpt_MDs_{self.MDsteps}_.pickle", 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # MMB open to clean-up
        if  self.MDsteps ==1:
            temp_file = open(f'{self.output_params.output_prefix}_system.dat','w')
            temp_file.close()

        # MMB CHANGE -- Print only on specific MD steps
        if i % self.output_params.energy_increment==0 or i ==0:

            test_mesh = trimesh.Trimesh(vertices=self.mesh.x, faces=self.mesh.f)
            mesh_volume = test_mesh.volume
            mesh_area   = test_mesh.area

            if self.print_bending_energy:
                bending_energy_temp = self.estore.properties(self.mesh.trimesh).bending

                with open(f'{self.output_params.output_prefix}_system.dat','a+') as f:
                    f.write(f'{i} {self.estore.energy(self.mesh.trimesh)} {self.acceptance_rate} {mesh_volume} {mesh_area} {bending_energy_temp}\n')
                    f.flush()
            else:
                with open(f'{self.output_params.output_prefix}_system.dat','a+') as f:
                    f.write(f'{i} {self.estore.energy(self.mesh.trimesh)} {self.acceptance_rate} {mesh_volume} {mesh_area} 0\n')
                    f.flush()


        if self.output_params.info and (i % self.output_params.info == 0):
            self.timer.timestamps.append(time.time())
            if len(self.timer.timestamps) == 2:
                tspan = self.timer.timestamps[1] - self.timer.timestamps[0]
                speed = tspan / self.output_params.info
                finish = self.timer.start + timedelta(seconds=tspan) * self.n
                print("\n-- Performance measurements")
                print(f"----- estimated speed: {speed:.3e} s/step")
                print(f"----- estimated end:   {finish}")
                self.timer.timestamps.pop(0)

            # Section for the preformance measurement of the code
        if i == 1:
            with open(f'{self.output_params.output_prefix}_performance.dat', 'w') as file:
                file.write(
                    '#Step Elapsed_Time Time_Per_Step %Vertex_Moves %Mesh_Flips %Residue %flip_att/num RAM_USAGE %RAM RAM_AVAILABLE_PRC RAM_TOTAL\n')
                file.flush()
                # tracemalloc.start()

        if (i % self.output_params.performance_increment == 0):
            self.timer.performance_timestamps.append(time.time())
            section_time = self.timer.timearray_new - self.timer.timearray
            self.timer.timearray = self.timer.timearray_new.copy()
            self.process = psutil.Process()

            if len(self.timer.performance_timestamps) == 2:
                performance_tspan = self.timer.performance_timestamps[1] - self.timer.performance_timestamps[0]

                fr=0.0
                if self.f_num!=0:
                    fr=self.f_att / self.f_num

                with open(f'{self.output_params.output_prefix}_performance.dat', 'a') as file:
                    file.write(f'{i} {self.timer.performance_timestamps[1] - self.timer.performance_start:.4f}'
                               f' {performance_tspan / self.output_params.performance_increment:.4f}'
                               f' {section_time[0] / performance_tspan:.4f} {section_time[1] / performance_tspan:.4f}'
                               f' {(performance_tspan - section_time[0] - section_time[1]) / performance_tspan:.4f}'
                               f' {fr:.4f} {self.process.memory_info().vms / 1024 ** 3:.4f}'
                               f' {self.process.memory_percent(memtype="vms"):.4f} {psutil.virtual_memory()[1] / 1000000000:.4f}'
                               f' {psutil.virtual_memory()[0] / 1000000000:.4f}\n'
                               )
                    file.flush()

                self.timer.performance_timestamps.pop(0)
                #{self.process.cpu_percent(interval=None): .4f}

    ############################################################################
    #              *SELF FUNCTIONS*: MINIMIZE HELFRICH HAMILTONIAN             #
    # MW: Preconditioning using standard trimem functionality                  #
    # See Trimem documentation for details                                     #
    ############################################################################

    def minim(self):
        """
        Run (precursor) minimization.

        Performs a minimization of the Helfrich bending energy as defined
        by the `config`.

        Args:
            mesh (mesh.Mesh): initial geometry.
            estore (EnergyManager): EnergyManager.
            config (dict-like): run-config file.

        """
        refresh_safe = self.algo_params.refresh

        if not self.algo_params.refresh == 1:
            wstr = f"SURFACEREPULSION::refresh is set to {self.algo_params.refresh}, " + \
                   "which is ignored in in minimization."
            warnings.warn(wstr)

            self.algo_params.refresh = 1

        step_count = Counter(move=0, flip=0)

        def _cb(x):
            self.callback(x, step_count)
            step_count["move"] += 1

        # run minimization
        options = {
            "maxiter": self.algo_params.maxiter,
            "disp": 0,
        }
        res = minimize(
            self.fun,
            self._ravel(self.mesh.x),
            #self.mesh.x,
            jac=self.grad,
            callback=_cb,
            method="L-BFGS-B",
            options=options
        )
        self.mesh.x = res.x.reshape(self.mesh.x.shape)
        self.algo_params.refresh=refresh_safe

        # print info
        print("\n-- Minimization finished at iteration", res.nit)
        print(res.message)
        self.estore.print_info(self.mesh.trimesh)

        # write final checkpoint
        self.cpt_writer()
        self.reset_counter()
        #self.reset_output_counter()

    ############################################################################
    #        *SELF FUNCTIONS*: CHECKPOINT CREATION - USES PICKLE               #
    ############################################################################

    def __reduce__(self):
        return self.__class__,(self.initialize,
                self.debug_mode,
                self.num_particle_types,
                self.mass_particle_type,
                self.group_particle_type,
                self.mesh.x,
                self.mesh.f,
                self.lmp.numpy.extract_atom('v')[:self.n_vertices,:],
                self.vertices_at_edge,
                self.estore.eparams.bond_params.type,
                self.estore.eparams.bond_params.r,
                self.estore.eparams.bond_params.lc0,
                self.estore.eparams.bond_params.lc1,
                self.estore.eparams.bond_params.a0,
                self.estore.eparams.repulse_params.n_search,
                self.estore.eparams.repulse_params.rlist,
                self.estore.eparams.repulse_params.exclusion_level,
                self.estore.eparams.repulse_params.lc1,
                self.estore.eparams.repulse_params.r,
                self.estore.eparams.continuation_params.delta,
                self.estore.eparams.continuation_params.lam,
                self.estore.eparams.kappa_b,
                self.estore.eparams.kappa_a,
                self.estore.eparams.kappa_v,
                self.estore.eparams.kappa_c,
                self.estore.eparams.kappa_t,
                self.estore.eparams.kappa_r,
                self.estore.eparams.area_frac,
                self.estore.eparams.volume_frac,
                self.estore.eparams.curvature_frac,
                self.print_bending_energy,
                self.algo_params.num_steps,
                self.algo_params.reinitialize_every,
                self.algo_params.init_step,
                self.algo_params.step_size,
                self.algo_params.traj_steps,
                self.algo_params.momentum_variance,
                self.algo_params.flip_ratio,
                self.algo_params.flip_type,
                self.algo_params.initial_temperature,
                self.algo_params.cooling_factor,
                self.algo_params.start_cooling,
                self.algo_params.maxiter,
                self.algo_params.refresh,
                self.algo_params.thermal_velocities,
                self.algo_params.pure_MD,
                self.algo_params.switch_mode,
                self.algo_params.box,
                self.periodic,
                self.equilibrated,
                self.equilibration_rounds,
                self.output_params.info,
                self.output_params.thin,
                self.output_params.out_every,
                self.output_params.input_set,  # hast to be stl file or if True uses mesh
                self.output_params.output_prefix,
                self.output_params.restart_prefix,
                self.output_params.checkpoint_every,
                self.output_params.output_format,
                self.output_params.output_flag,
                self.output_params.output_counter,
                self.output_params.performance_increment,
                self.output_params.energy_increment,

                self.estore.initial_props.area,
                self.estore.initial_props.volume,
                self.estore.initial_props.curvature,
                self.estore.initial_props.bending,
                self.estore.initial_props.tethering,
                #self.estore.initial_props.repulsion,

                self.timer.performance_start,
                self.timer.performance_timestamps,
                self.timer.timestamps,
                self.timer.timearray,
                self.timer.timearray_new,
                self.timer.start,
                self.counter["move"],
                self.counter["flip"],
                self.beads.n_beads,
                self.beads.n_bead_types,
                self.lmp.numpy.extract_atom('x')[self.n_vertices:,:],
                self.lmp.numpy.extract_atom('v')[self.n_vertices:, :],
                self.beads.bead_sizes,
                self.beads.types,
                self.n_bond_types,
                )
    
    def __getstate__(self):
        # This method is called when pickling, customize the state saved
        return (self.mesh_points, self.mesh_faces)

    def __setstate__(self, state):
        # This method is called when unpickling, customize how state is restored
        self.mesh_points, self.mesh_faces = state

    # checkpoints using pickle
    def make_checkpoint_handle(self):
        return self.make_checkpoint

    def make_checkpoint(self, force_name=None):

        if not force_name:

            cptfname = pathlib.Path(self.output_params.output_prefix)
            cptfname = cptfname.name + self.output_params.output_flag + '.cpt'

            with open(cptfname, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
                #json.dump(self, f)
            if self.output_params.output_flag == 'A':
                self.output_params.output_flag = 'B'
            else:
                self.output_params.output_flag = 'A'
        else:
            cptfname = pathlib.Path(force_name)
            with open(cptfname, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
                #json.dump(self, f)
        print(f'made cp:{cptfname}')

    # SOME UTILITY FUNCTIONS
    # Here we have some minor utility functions to set
    # parameters, counters, ....

    def extra_callback(self, timearray_loc):
        self.timer.timearray_new=timearray_loc

    def update_energy_manager(self):
        self.estore = m.EnergyManager(self.mesh.trimesh, self.eparams)

    def update_energy_parameters(self):
        self.eparams = self.estore.eparams

    def reset_counter(self,move=0,flip=0):

        self.counter = Counter(move=move, flip=flip)

    def reset_output_counter(self):
        self.output_params.output_counter=0

    def update_output_counter(self,ocn):
        self.output_params.output_counter = ocn

    def update_output(self):
        self.output = make_output(self.output_params.output_format, self.output_params.output_prefix,
                                  self.output_params.output_counter, callback=self.update_output_counter)

    @staticmethod
    def parse_boundary(periodic: pyUnion[bool, list[bool]]) -> str:
        if isinstance(periodic, bool):
            return "p p p" if periodic else "f f f"
        elif isinstance(periodic, list) and len(periodic) == 3:
            return " ".join("p" if p else "f" for p in periodic)
        else:
            raise ValueError("periodic must be a bool or a list of 3 bools")
        
################################################################################
#                       READ AND LOAD CHECKPOINTS                              #
################################################################################

def read_checkpoint(fname):

    with open(fname, 'rb') as f:
        trilmp = pickle.load(f)
    return trilmp

def load_checkpoint(name, alt='last'):

    if alt=='last':
        cp = ['A', 'B']
        n = []

        for c in cp:
            trilmp = read_checkpoint(f'{name}{c}.cpt')
            n.append(sum(trilmp.counter.values()))

        if n[0] > n[1]:
            trilmp = read_checkpoint(f'{name}A.cpt')
            print('reading A')
        else:
            trilmp = read_checkpoint(f'{name}B.cpt')
            print('reading B')

    if alt=='explicit':
        trilmp = read_checkpoint(f'{name}')
    else:
        trilmp = read_checkpoint(f'{name}.cpt')

    return trilmp
