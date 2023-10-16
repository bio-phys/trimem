
import re,textwrap
from typing import Sequence
_sp = u'\U0001f604'
_nl = '\n'+_sp
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



import warnings
from datetime import datetime, timedelta
import psutil
import os

from copy import copy

import trimesh
from .. import core as m
from trimem.core import TriMesh
from trimem.mc.mesh import Mesh

from trimem.mc.output import make_output
from collections import Counter

import pickle
import pathlib

import numpy as np
import time
from scipy.optimize import minimize

import trimem.mc.trilmp_h5






from ctypes import *
from lammps import lammps, PyLammps, LAMMPS_INT, LMP_STYLE_GLOBAL, LMP_VAR_EQUAL, LMP_VAR_ATOM, LMP_TYPE_SCALAR, LMP_TYPE_VECTOR, LMP_TYPE_ARRAY, LMP_SIZE_VECTOR, LMP_SIZE_ROWS, LMP_SIZE_COLS


###############################################################################################
"""
TriLmp - Combinig TRIMEM and LAMMPS to enable versatile MD simulations using triangulated mesh membranes

This program was created in 2023 by Michael Wassermair in the course of an internship in the Saric Group (ISTA).
It is based on the software TRIMEM, which was originally intended to perform
HMC energy minimization of the Helfrich Hamiltonian using a vertex-averaged triangulared mesh discretisation.
By connecting the latter with LAMMPS we expose the mesh vertices defining the membrane to interactions with external
particles using LAMMPS pair_ or bond_styles.

It is dependent on a modified version of trimem and LAMMPS using specific packages 
and the additional pair_styles nonreciprocal and nonreciprocal/omp 
See SETUP_GUIDE.txt for details.

The overall structure of the programm is 

-Internal classes used by TriLmp
-TriLmp Object

-- Default Parameters -> description of all parameters
-- Initialization of Internal classes using parameters
-- Init. of TRIMEM EnergyManager used for gradient/energy of helfrich hamiltonian

-- LAMMPS initialisation
--- Creating instances of lmp
--- Setting up Basic system
--- calling Lammps functions to set thermostat and interactions

-- FLIPPING Functions  
-- HMC/MD Functions + Wrapper Functions used on TRIMEM side
-- RUN Functions -> simulation utility to be used in script
-- CALLBACK and OUTPUT 
-- Some Utility Functions
-- Minimize Function -> GD for preconditionining states for HMC
-- Pickle + Checkpoint Utility
-- LAMMPS scrips used for setup


"""
########################################################################
#                   INTERNAL CLASSES TO BE USED BY TriLmP              #
########################################################################

class Timer():
    "Storage for timer state to reinitialize PerformanceEnergyEvaluator after Reset"
    def __init__(self,ptime,ts,ta,tan,ts_default,stime):
        self.performance_start=ptime
        self.performance_timestamps=ts
        self.timearray=ta
        self.timearray_new=tan
        self.timestamps=ts_default
        self.start=stime


class InitialState():
    def __init__(self,area,volume,curvature,bending,tethering):
        """Storage for reference properties to reinitialize Estore after Reset. In case of reinstating Surface
        Repulsion in TRIMEM a repulsion property would have to be added again """
        self.area=area
        self.volume=volume
        self.curvature=curvature
        self.bending=bending
        self.tethering=tethering
        
class Beads():
    def __init__(self,n_types,bead_int,bead_int_params,bead_pos,bead_vel,bead_sizes,bead_masses,bead_types,self_interaction,self_interaction_params):
        """Storage for Bead parameters.

        Args:
            n_types: number of different types of beads used
            bead_int: interaction type ('lj/cut','nonreciprocal','tether') -> will be generalized
            bead_int_params: (NP,n_types) tuple of parameters used for interaction where NP is the number
                of used parameters and n_types the number of bead types,
                e.g. ((par1_bead1,par2_bead1),(par1_bead2,par2_bead2))
            bead_pos: (N,3) array of bead positions with N being
                the number of beads.
            bead_vel: (N,3) array of bead velocities with N being
                the number of beads.
            bead_sizes: (n_types,1) tuple containing the sizes of the beads, e.g. (size_bead1) or (size_bead1,size_bead2)
            bead_masses: (n_types,1) tuple containing the mass of the beads, for n_type==1 just use single float
            bead_types: (N_beads,1) tuple or array (must use 1 index) of the different types of the beads.
                Bead types are strictly >=2, e.g. 3 beads and 2 n_types (2,3,3)
            self_interaction: bool (default False) sets the same potential as used before for interaction of bead types
            self_interaction_params: same as interaction params but for the interaction of beads with beads of their own type
                -> TODO: interface to set all bead-bead interactions not yet implemented

            args: ignored

        Keyword Args:
            kwargs: ignored

        """
        self.n_beads=bead_pos.shape[0]
        self.n_types=n_types
        self.positions=bead_pos
        self.velocities=bead_vel
        self.types=bead_types
        self.masses=bead_masses
        self.bead_interaction=bead_int
        self.bead_interaction_params=bead_int_params
        self.bead_sizes=bead_sizes                      ## diameter
        self.self_interaction=self_interaction
        self.self_interaction_params=self_interaction_params


class OutputParams():
    """Containter for parameters related to the output option """
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
    """Containter for parameters related to the algorithms used  -> See DEFAULT PARAMETERS section for description"""
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
                 langevin_thermostat,
                 langevin_damp,
                 langevin_seed,
                 pure_MD,
                 switch_mode,
                 box,
                 additional_command
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
        self.langevin_thermostat=langevin_thermostat
        self.langevin_damp=langevin_damp
        self.langevin_seed=langevin_seed
        self.pure_MD=pure_MD
        self.switch_mode=switch_mode
        self.box=box
        self.additional_command=additional_command



###############################################################
#                  MAIN TRILMP CLASS OBJECT                   #
###############################################################

class TriLmp():

    def __init__(self,
                 ##############################################
                 #             DEFAULT PARAMETERS             #
                 ##############################################
                 #Initialization
                 initialize=True,     # Determines if mesh is used as new reference in estore
                 #MESH
                 mesh_points=None,    # positions of membrane vertices
                 mesh_faces=None,     # faces defining mesh
                 mesh_velocity=None,  # initial velocities
                 #BOND
                 bond_type='Edge',      # 'Edge' or 'Area
                 bond_r=2,              # steepness of potential walls
                 lc0=None,              # upper onset ('Edge') default will be set below
                 lc1=None,              # lower onset ('Edge')
                 a0=None,               # reference face area ('Area')
                 #SURFACEREPULSION
                 n_search="cell-list",  # neighbour list types ('cell and verlet') -> NOT USED TODO
                 rlist=0.1,             # neighbour list cutoff -> NOT USED TODO
                 exclusion_level=2,     # neighbourhood exclusion setting -> NOT IMPLEMENTED TODO
                 rep_lc1=None,           # lower onset for surface repusion (default set below)
                 rep_r= 2,              # steepness for repulsion potential
                 delta= 0.0,            # "timestep" parameter continuation for hamiltonian
                 lam= 1.0,              # cross-over parameter continuation for hamiltonian (see trimem -> NOT FULLY FUNCTIONAL YET)
                 kappa_b = 30.0,        # bending
                 kappa_a = 1.0e6,       # area penalty
                 kappa_v = 1.0e6,       # volume penalty
                 kappa_c = 1.0e6,       # curvature (area-difference) penalty
                 kappa_t = 1.0e5,       # tethering
                 kappa_r = 1.0e3,       # surface repulsion
                 area_frac = 1.0,       # target area (fraction of referrence)
                 volume_frac = 1.0,     # target volume
                 curvature_frac = 1.0,  # target curvature

                 #ALGORITHM
                 num_steps=10,                  # number of overall simulation steps (for trilmp.run() but overitten by trilmp.run(N))
                 reinitialize_every=10000,      # NOT USED TODO
                 init_step='{}',                # NOT USED TODO
                 step_size=7e-5,                # Trajectory length for each MD step
                 traj_steps=100,                # timestep for MD (fix nve)
                 momentum_variance=1.0,         # mass of membrane vertices
                 flip_ratio=0.1,                # fraction of flips intended to flip
                 flip_type='parallel',          # 'serial' or 'parallel'
                 initial_temperature=1.0,       # temperature of system
                 cooling_factor=1.0e-4,         # cooling factor for simulated anneadling -> NOT IMPLEMENTED TODO
                 start_cooling=0,               # sim step at which cooling starts -> NOT IMPLEMENTED TODO
                 maxiter=10,                    # parameter used for minimize function (maximum gradient steps
                 refresh=1,                     # refresh rate of neighbour list -> NOT USED TODO

                 #  !!!!!!!!!!!!!!!!!!!!!!!!!!!
                 ##############################
                 #  SIMULATION STYLE          #
                 ##############################
                 # !!!!!!!!!!!!!!!!!!!!!!!!!!!!
                 #################################################################################################
                 # to perform BD simulation (thermal_velocities=False, langevin_thermostat=True, pure_MD=True)   #
                 # to perform HMC simulation (thermal_velocities=True, langevin_thermostat=False, pure_MD=False) #
                 #################################################################################################
                 thermal_velocities=False,      # thermal reset of velocities at the begin of each MD step
                 langevin_thermostat=True,      # uses langevin thermostat to
                 langevin_damp=0.03,            # damping time for thermostat
                 langevin_seed=1,               # seed for thermostat noise
                 pure_MD=True,                  # accept every MD trajectory
                 switch_mode='random',          # 'random' or 'alernating' -> either random choice of flip-or-move
                 box=(-10,10,-10,10,-10,10),    # Simulation BOX (periodic)
                 additional_command=None,        # add a LAMMPS command """ COMMAND 1
                                                #                          COMMAND 2  ...    """ at the end of LAMMPS initialisation



                 #OUTPUT
                 info=10,              # output hmc and flip info to shell every nth step
                 thin=10,              # output trajectory every nth step
                 out_every= 0,         # output minimize state every nth step (only for gradient minimization)

                 input_set='inp.stl',  # hast to be stl file or if True uses mesh -> NOT USED TODO
                 output_prefix='inp',  # name of simulation files
                 restart_prefix='inp', # name for checkpoint files
                 checkpoint_every= 1,  # checkpoint every nth step
                 output_format='xyz',  # output format for trajectory -> NOT YET (STATISFYINGLY) IMPLEMENTED TODO

                 output_flag='A',      # initial flag for output (alternating A/B)
                 output_counter=0,     # used to initialize (outputted) trajectory number in writer classes
                 performance_increment=1000,  # print performance stats every nth step to output_prefix_performance.dat
                 energy_increment=250,        # print total energy to energy.dat

                # REFERENCE STATE DATA
                 area=1.0,                    # placeholders to be filled by reference state parameters (used to reinitialize estore)
                 volume=1.0,
                 curvature=1.0,
                 bending=1.0,
                 tethering=1.0,
                 #repulsion=1.0,             # would be used if surface repulsion was handeld by TRIMEM
                # TIMING UTILITY
                 ptime=time.time(),  #
                 ptimestamp=[],      # used to time performence stats
                 dtimestamp=[],
                 timearray=np.zeros(2),
                 timearray_new=np.zeros(2),
                 stime=datetime.now(),
                # COUNTERS
                 move_count=0,   #move counts (to initialize with)
                 flip_count=0,   #flip counts (to initialize with) -> use this setting to set a start step (move+flip)


                 # BEADS (see Bead class for details)
                 n_types=0,
                 bead_int='lj/cut',
                 bead_int_params=(0,0),
                 bead_pos=np.zeros((0,0)),
                 bead_vel=None,
                 bead_sizes=0.0,
                 bead_masses=1.0,
                 bead_types=[],
                 self_interaction=False,
                 self_interaction_params=(0,0)
                 ):

        
       ##################################
       #    SOME MINOR PREREQUESITES    #
       ##################################

        # initialize using mesh arguments?
        self.initialize = initialize
        self.acceptance_rate = 0.0
        # used for minim
        self.flatten = True
        if self.flatten:
            self._ravel = lambda x: np.ravel(x)
        else:
            self._ravel = lambda x: x

        # different bond types  (used for tether potential in TRIMEM)
        self._bond_enums = {
            "Edge": m.BondType.Edge,
            "Area": m.BondType.Area
        }


        #########################
        #         MESH          #
        #########################
        # Argument: mesh should be Mesh object gets converted to Mesh.trimesh (TriMesh) internally
        self.mesh = Mesh(points=mesh_points, cells=mesh_faces)
        if pure_MD:
            self.mesh_temp=0
        else:
            self.mesh_temp = Mesh(points=mesh_points, cells=mesh_faces)
        self.mesh_velocity=mesh_velocity
        self.n_vertices=self.mesh.x.shape[0]

        #######################
        #        BEADS        #
        #######################
        self.beads=Beads(n_types,
                         bead_int,
                         bead_int_params,
                         bead_pos,
                         bead_vel,
                         bead_sizes,
                         bead_masses,
                         bead_types,
                         self_interaction,
                         self_interaction_params)


        #######################
        #   BONDS/TETHERS     #
        #######################
        self.bparams = m.BondParams()
        if issubclass(type(bond_type), str):
            self.bparams.type = self._bond_enums[bond_type]
        else:
            self.bparams.type=bond_type
        self.bparams.r = bond_r

        # default setting for bonds related to average distance of initial mesh
        ###############################################
        # !! MEMBRANE LENGTHSCALES ARE SET HERE  !!   #
       ################################################
        if (lc1 is None) and (lc0 is None) and self.initialize:
            a, l = m.avg_tri_props(self.mesh.trimesh)
            self.bparams.lc0 = 1.25 * l
            self.bparams.lc1 = 0.75 * l
            self.bparams.a0 = a
        else:
            self.bparams.lc0 = lc0
            self.bparams.lc1 = lc1
            self.bparams.a0  = a0


        print(f'l0={self.bparams.lc0/1.25}\nlc1={self.bparams.lc1}\nlc0={self.bparams.lc0}\n')


        #################################
        #      SURFACE REPULSION        #
        #################################
        #      handeled by LAMMPS       #
        #################################

        self.rparams = m.SurfaceRepulsionParams()
        # neighbour lists of TRIMEM are not used but kept here in case needed
        self.rparams.n_search = n_search
        self.rparams.rlist = rlist
        #

        #currently default 2 is fixed, not yet implemented to change TODO
        self.rparams.exclusion_level = exclusion_level

        if rep_lc1==None:
            #by default set to average distance used for scaling tether potential
            self.rparams.lc1 = self.bparams.lc1/0.75 #rep_lc1
        else:
            self.rparams.lc1=rep_lc1
        #self.rparams.lc1 = l*0.001
        self.rparams.r = rep_r

        print(f'l0={self.bparams.lc0 / 1.25}\nlc1={self.bparams.lc1}\nlc0={self.bparams.lc0}\nr_lc0={self.rparams.lc1}\n')
        ###############################
        #  PARAMETER CONTINUATION     #
        ###############################
        # translate energy params
        # see trimem doc for details
        ###############################
        self.cp = m.ContinuationParams()
        self.cp.delta = delta
        self.cp.lam = lam

        ###############################
        #     ENERGY PARAMETERS       #
        ##################################################################
        # CAVEAT!! If one wants to change parameters between             #
        # two trilmp.run() commands in a simulation script               #
        # one has to alter trilmp.estore.eparams.(PARAMETER OF CHOICE).  #
        # These parameters will also be used to reinitialize LAMMPS, i.e #
        # in pickling for the checkpoints                                #
        ##################################################################
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

       ###############################
       #   ALGORITHMIC PARAMETERS    #
       ###############################
        self.algo_params=AlgoParams(num_steps,reinitialize_every,init_step,step_size,traj_steps,
                 momentum_variance,flip_ratio,flip_type,initial_temperature,
                 cooling_factor,start_cooling,maxiter,refresh,thermal_velocities,
                langevin_thermostat,
                langevin_damp,
                langevin_seed,
                pure_MD,
                switch_mode,
                box,additional_command)

       ###############################
       #     OUTPUT PARAMETERS       #
       ###############################
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

        ####################################################
        #     ENERGY MANAGER INITIALISATION - TRIMEM       #
        ####################################################
        if self.initialize:
            # setup energy manager with initial mesh
            self.estore = m.EnergyManagerNSR(self.mesh.trimesh, self.eparams)
            #safe initial states property
            self.initial_state=InitialState(self.estore.initial_props.area,
                                            self.estore.initial_props.volume,
                                            self.estore.initial_props.curvature,
                                            self.estore.initial_props.bending,
                                            self.estore.initial_props.tethering)
                                            #self.estore.initial_props.repulsion)

            self.initialize=False

        else:
            # reinitialize using saved initial state properties (for reference potential V, A, dA)
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

            #recreate energy manager
            self.estore = m.EnergyManagerNSR(self.mesh.trimesh, self.eparams, self.init_props)
       # save general lengthscale, i.e. membrane bead "size" defined by tether repulsion onset
        self.l0 = self.estore.eparams.bond_params.lc1 / 0.75

    ########################################
    #         MASSES OF PARTICLES          #
    ########################################
    # set Mass Vector (vector of all masses (vertices + beads)
        self.masses = []
        for i in range(self.n_vertices):
            self.masses.append(self.algo_params.momentum_variance)

        if self.beads.n_beads:
            if self.beads.n_types > 1:
                for i in range(self.beads.n_beads):
                    self.masses.append(self.beads.masses[self.beads.types[i]-2])
            else:
                for i in range(self.beads.n_beads):
                    self.masses.append(self.beads.masses)
        self.masses=np.asarray(self.masses)



    ##############################################################
    #                     LAMMPS SETUP                           #
    ##############################################################


        # create internal lammps instance
        self.lmp = lammps(cmdargs=['-sf','omp'])
        self.L = PyLammps(ptr=self.lmp,verbose=False)

        ###########################################
        #            Optional Tethers             #
        ###########################################
        #default
        bond_text="""
                    special_bonds lj/coul 0.0 0.0 0.0
                    bond_style zero nocoeff
                    bond_coeff * * 0.0  """
        n_bond_types=1
        n_tethers=0
        add_tether=False
        #added tethers
        if self.beads.bead_interaction=='tether':
            add_tether=True
            n_tethers=self.beads.n_beads
            n_bond_types=2
            bond_text=f"""
            special_bonds lj/coul 0.0 0.0 0.0
            bond_style hybrid zero harmonic
            bond_coeff 1 zero 0.0
            bond_coeff 2 harmonic {self.beads.bead_interaction_params[0]} {self.beads.bead_interaction_params[1]}
            special_bonds lj/coul 0.0 0.0 0.0
            
            """

        ############################################################
        #        MAIN INPUT SCRIPT USED IN LAMMPS                  #
        ############################################################
        #  For significant changes to functionality                #
        #  this section might have to be altered                   #
        #  Further LAMMPS scipts used to setup lammps are          #
        #  defined as function in the end of this class definition.#
        #  CAVEAT: Currently all the scripts and the callback to   #
        #  TRIMEM are critically reliant on the the setting:       #
        # " atom_modify sort 0 0.0 "                               #
        ############################################################
        basic_system = dedent(f"""\
            units lj
            dimension 3
            package omp 0
            
            atom_style	hybrid bond charge 
            atom_modify sort 0 0.0
            

            region box block {self.algo_params.box[0]} {self.algo_params.box[1]} {self.algo_params.box[2]} {self.algo_params.box[3]} {self.algo_params.box[4]} {self.algo_params.box[5]}
            create_box {1+self.beads.n_types} box bond/types {n_bond_types} extra/bond/per/atom 14 extra/special/per/atom 14
                
            run_style verlet
            fix 1 all  nve
            fix ext all external pf/callback 1 1
            
            timestep {self.algo_params.step_size}

            {block(bond_text)}            
            
            dielectric  1.0
            compute th_ke all ke
            compute th_pe all pe pair bond
            
            thermo {self.algo_params.traj_steps}                                        
            thermo_style custom c_th_pe c_th_ke
            thermo_modify norm no
            
            info styles compute out log

            echo log
            log none
            
        """)

        # initialize lammps
        self.lmp.commands_string(basic_system)


        


        # VERTICES + TOPOLOGY + BEADS + TETHERS    

        # extract bond topology from Trimesh object
        self.edges = trimesh.Trimesh(vertices=self.mesh.x, faces=self.mesh.f).edges_unique
        self.edges=np.unique(self.edges,axis=0)
        self.n_edges=self.edges.shape[0]

        with open('sim_setup.in', 'w') as f:
            f.write('\n\n')

            f.write(f'{self.mesh.x.shape[0]+self.beads.n_beads} atoms\n')
            f.write(f'{self.edges.shape[0]+n_tethers} bonds\n\n')

            f.write(f'{1+self.beads.n_types} atom types\n')
            if add_tether:
                f.write(f'2 bond types\n\n')
            else:
                f.write(f'1 bond types\n\n')

          #  f.write('Bond Coeffs\n\n')
          #  f.write(f'1 zero nocoeff\n')
          #  if add_tether:
          #      f.write(f'2 harmonic {self.beads.bead_interaction_params[0]} {self.beads.bead_interaction_params[1]}\n')


            f.write('Masses\n\n')
            f.write(f'1 {self.algo_params.momentum_variance}\n')
            if self.beads.n_beads:
                for i in range(self.beads.n_types):
                    if self.beads.n_types>1:

                        f.write(f'{i+2} {self.beads.masses[i]}\n')
                    else:
                        f.write(f'{i + 2} {self.beads.masses}\n')


            f.write(f'Atoms # hybrid\n\n')
            for i in range(self.n_vertices):
                f.write(f'{i + 1} 1  {self.mesh.x[i, 0]} {self.mesh.x[i, 1]} {self.mesh.x[i, 2]} 1 1.0 \n')

            if self.beads.n_beads:
                if self.beads.n_types>1:
                    for i in range(self.beads.n_beads):
                        f.write(f'{self.n_vertices+1+i} {self.beads.types[i]} {self.beads.positions[i,0]} {self.beads.positions[i,1]} {self.beads.positions[i,2]} 1 1.0\n')
                else:
                    for i in range(self.beads.n_beads):
                        f.write(f'{self.n_vertices+1+i} 2 {self.beads.positions[i,0]} {self.beads.positions[i,1]} {self.beads.positions[i,2]} 1 1.0\n')


            # f.write(f'{self.mesh.x.shape[0]+1} {self.mesh.x[0,0]} {self.mesh.x[0,1]} {self.mesh.x[0,2]} 1 2\n')
            f.write(f'Bonds # zero special\n\n')

            for i in range(self.edges.shape[0]):
                f.write(f'{i + 1} 1 {self.edges[i, 0] + 1} {self.edges[i, 1] + 1}\n')

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


        self.lmp.commands_string(self.pair_cmds())

        if self.beads.n_beads:
            self.lmp.commands_string("neigh_modify one 5000 page 50000 every 100 check yes")

        # initialize LAMMPS
        self.lmp.command('read_data sim_setup.in add merge')


        # THERMOSTAT/VELOCITIES SETTINGS            #
        self.atom_props = f"""     
                        velocity vertices create {self.algo_params.initial_temperature} 1298371 mom yes dist gaussian 
                        """

        #group together different types
        self.lmp.command('group vertices type 1')
        for i in range(self.beads.n_types):
            self.lmp.command(f'group beads_{i+2} type {i+2}')

        # Langevin thermostat for all beads scaling each particle type according to it's size and mass (M/sigma)
        if self.algo_params.langevin_thermostat:
            sc0=self.algo_params.momentum_variance/self.l0
            bds = ''
            if self.beads.n_types:
                if self.beads.n_types>1:
                    bds = ''
                    for i in range(self.beads.n_types):
                        bds += f'scale {i + 2} {(self.beads.masses[i] / self.beads.bead_sizes[i])/sc0} '
                else:
                    bds=f'scale 2 {(self.beads.masses / self.beads.bead_sizes)/sc0} '

            lv_thermo_comm=f"""
                            fix lvt all langevin {self.algo_params.initial_temperature} {self.algo_params.initial_temperature}  {self.algo_params.langevin_damp} {self.algo_params.langevin_seed} zero yes {bds}
                            
                            """
            self.lmp.commands_string(lv_thermo_comm)


        # initialize random velocities if thermal velocities is chosen or set to 0
        if self.algo_params.thermal_velocities:
            self.lmp.commands_string(self.atom_props)
        else:
            self.lmp.command('velocity all zero linear')


        # setting or reinitializing bead velocities
        if np.any(self.beads.velocities):
            for i in range(self.n_vertices,self.n_vertices+self.beads.n_beads):
                self.L.atoms[i].velocity=self.beads.velocities[i-self.n_vertices,:]

        # setting or reinitializing mesh velocities
        if np.any(self.mesh_velocity):
            for i in range(self.n_vertices):
                self.L.atoms[i].velocity=self.mesh_velocity[i,:]
        
        # INTERACTIONS  and TRIMEM callback
        
        # set callback for helfrich gradient to be handed from TRIMEM to LAMMPS via fix external "ext"
        self.lmp.set_fix_external_callback("ext", self.callback_one, self.lmp)

        if self.algo_params.additional_command:
            self.lmp.commands_string(self.algo_params.additional_command)


        # set temperature TODO: no implementation of the simulated annealing for MD mode yet
        # Temperature in LAMMPS set to fixed initial temperature
        self.T = self.algo_params.initial_temperature

        # approx. initialization of energy components (i.e. first step for HMC will probably be accepted, irrelevant for pureMD)
        v=self.lmp.numpy.extract_atom("v")
        self.pe=0.0
        self.ke=0.5 * self.algo_params.momentum_variance*v.ravel().dot(v.ravel())
        self.he=self.estore.energy(self.mesh.trimesh)#+0.5 * v.ravel().dot(v.ravel())
        self.energy_new=0.0


        #setting and getting helffrich energy from/to lammps
        #self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))
        #print(self.lmp.numpy.extract_fix("ext", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR, nrow=0))
        
        # BOOKKEEPING
        # Setting up counters for stats and writers

        # flip stats
        self.f_i = 0
        self.f_acc = 0
        self.f_num = 0
        self.f_att = 0

        # move stats
        self.m_i = 0
        self.m_acc = 0

        self.counter = Counter(move=move_count, flip=flip_count)
        self.timer = Timer(ptime, ptimestamp, timearray, timearray_new,dtimestamp,stime)

        self.cpt_writer = self.make_checkpoint_handle()
        self.process=psutil.Process()
        self.n= self.algo_params.num_steps // self.output_params.info if self.output_params.info!=0 else 0.0


        self.info_step = max(self.output_params.info, 0)
        self.out_step = max(self.output_params.thin, 0)
        self.cpt_step = max(self.output_params.checkpoint_every, 0)
        self.refresh_step = max(self.algo_params.refresh, 0)

       #####                   TRAJECTORY WRITER SETTINGS                        #####
       ###############################################################################
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

       #############################################################################
        #####                           FLIPPING                                #####
        #############################################################################
        #  In this section the serial or parallel flipping method is chosen         #
        #  and functions forwarding the updated topology to LAMMPS are defined      #
        #  we use the flip function flip_nsr/pflip_nsr which are reliant on the     #
        #  the use of the estore_nsr. Hence we shut off the calculation of surface  #
        #  repusion in TRIMEM. If for some reason this functionality should be n    #
        #  needed one would have to remove all '_nsr' suffixes and use the          #
        #  estore.eparams.repulse_params which are kept for backwards portability   #
        #############################################################################

        # chosing function to be used for flipping
        if self.algo_params.flip_type == "none" or self.algo_params.flip_ratio == 0.0:
            self._flips = lambda: 0
        elif self.algo_params.flip_type == "serial":
            self._flips = lambda: m.flip_nsr(self.mesh.trimesh, self.estore, self.algo_params.flip_ratio)
        elif self.algo_params.flip_type == "parallel":
            self._flips = lambda: m.pflip_nsr(self.mesh.trimesh, self.estore, self.algo_params.flip_ratio)
        else:
            raise ValueError("Wrong flip-type: {}".format(self.algo_params.flip_type))

    # test function to perform a single flip in lammps from (i,j)-> (k,l)
    def lmp_flip_single(self,i,j,k,l):

        self.lmp.command(f'group flip_off id {i} {j}')
        self.lmp.command('delete_bonds flip_off bond 1 remove')
        self.lmp.command('group flip_off clear')
        self.lmp.command(f'create_bonds single/bond 1 {k} {l}')

    # function used for flipping
    def lmp_flip(self,flip_id):

        nf=flip_id[-1][0]

        if nf:

            del_com='remove'

            for i in range(nf):
                if i == nf-1:
                    del_com = 'remove special'

                self.lmp.command(f'create_bonds single/bond 1 {flip_id[i][0] + 1} {flip_id[i][1] + 1}')
                self.lmp.command(f'group flip_off id {flip_id[i][2] + 1} {flip_id[i][3] + 1}')
                self.lmp.command(f'delete_bonds flip_off bond 1 {del_com}')
                self.lmp.command('group flip_off clear')

                    #self.lmp.command(f'delete_bonds flip_off bond 1 special')

                #ids+=f'{flip_id[i*4+2]+1} {flip_id[i*4+3]+1} '
           # print(ids)
           # self.lmp.command(f'group flip_off id {ids}')
           # self.lmp.command('delete_bonds flip_off bond 1 remove special')
           # self.lmp.command('group flip_off clear')
           # for i in range(nf):
           #     self.lmp.command(f'create_bonds single/bond 1 {flip_id[i * 4] + 1} {flip_id[i * 4 + 1] + 1} special yes')
        else:
            pass




    def flip_info(self):
        """Print algorithmic information."""
        i_total = sum(self.counter.values())
        if self.output_params.info and i_total % self.output_params.info == 0:
            # MMB CHANGED -- MAKING SURE THAT NUMBER OF EDGES KNOWS ABOUT THE FLIP RATIO
            n_edges = self.mesh.trimesh.n_edges()*self.algo_params.flip_ratio
            ar      = self.f_acc / (self.f_i * n_edges) if not self.f_i == 0 else 0.0
            self.acceptance_rate = ar
            #print("Accepted", self.f_acc)
            #print("Number of candidate edges", n_edges)
            
            print("\n-- MCFlips-Step ", self.counter["flip"])
            print("----- flip-accept: ", ar)
            print("----- flip-rate:   ", self.algo_params.flip_ratio)
            self.f_acc = 0
            self.f_i   = 0
            self.f_num = 0
            self.f_att = 0

    def flip_step(self):
        """Make one step."""
        flip_ids=self._flips()
        self.mesh.f[:]=self.mesh.f
        #print(flip_ids)
        self.lmp_flip(flip_ids)


        self.f_acc += flip_ids[-1][0]
        self.f_num += flip_ids[-1][1]
        self.f_att += flip_ids[-1][2]
        self.f_i += 1
        self.counter["flip"] += 1


    ################################################################
    #                 MOLECULAR DYNAMICS / HMC                     #
    ################################################################
    # In this section we combine all functions that are used for   #
    # either HMC/pure_MD run or minimize including the wrapper     #
    # functions used for updating the mesh (on the TRIMEM side)    #
    # when evaluating the gradient                                 #
    ################################################################



    def hmc_step(self):
        if not self.algo_params.pure_MD:

            # setting temperature
            i = sum(self.counter.values())
            Tn = np.exp(-self.algo_params.cooling_factor * (i - self.algo_params.start_cooling)) * self.algo_params.initial_temperature
            self.T = max(min(Tn, self.algo_params.initial_temperature), 10e-4)

            # safe mesh for reset in case of rejection
            self.mesh_temp=copy(self.mesh.x)
            self.beads_temp=copy(self.beads.positions)

            #calute energy
            #future make a flag system to avoid double calculation if lst step was also hmc step
            if self.algo_params.thermal_velocities:
                 self.atom_props = f"""     
                            
                            velocity        vertices create {self.T} {np.random.randint(1,9999999)} mom yes dist gaussian   
    
                            """
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

        else:
            self.lmp.command(f'run {self.algo_params.traj_steps} post no')
            self.m_acc += 1
            self.m_i += 1

            self.counter["move"] += 1


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

    ##################
    #      RUN!      #
    ###################################################
    #       COMBINED MOVE FOR SIMULATION              #
    ###################################################
    def step_random(self):

        """Make one step each with each algorithm."""
        if np.random.choice(2) == 0:
            t_fix = time.time()
            self.hmc_step()
            self.timer.timearray_new[0] += (time.time() - t_fix)

        else:
            t_fix = time.time()
            self.flip_step()
            self.timer.timearray_new[1] += (time.time() - t_fix)

    def step_alternate(self):

        """Make one step each with each algorithm."""

        t_fix = time.time()
        self.hmc_step()
        self.timer.timearray_new[0] += (time.time() - t_fix)
        t_fix = time.time()
        self.flip_step()
        self.timer.timearray_new[1] += (time.time() - t_fix)


    def run(self,N=0):
        if N==0:
            N=self.algo_params.num_steps

        if self.algo_params.switch_mode=='random':
            self.step = lambda: self.step_random()
            sim_steps=N
        elif self.algo_params.switch_mode=='alternating':
            self.step = lambda: self.step_alternate()
            sim_steps=np.int64(np.floor(N/2))
        else:
            raise ValueError("Wrong switchmode: {}. Use 'random' or 'alternating' ".format(self.algo_params.flip_type))

        for i in range(sim_steps):
            self.step()
            self.hmc_info()
            self.callback(np.copy(self.mesh.x),self.counter)
            self.flip_info()


    ################################################
    #         WRAPPER FUNCTIONS                    #
    ################################################

    # Decorators for meshupdates when calling force function
    def _update_mesh(func):
        """VARIANT FOR USE WITH self.minim():Decorates a method with an update of the mesh vertices.

        The method must have signature f(self, x, \*args, \*\*kwargs) with
        x being the new vertex coordinates.
        """
        def wrap(self, x, *args, **kwargs):
            self.mesh.x = x.reshape(self.mesh.x.shape)
            return func(self, x, *args, **kwargs)
        wrap.__doc__  = func.__doc__
        wrap.__name__ = func.__name__
        return wrap


    def _update_mesh_one(func):
        """VARIANT FOR USE WITH LAMMPS: Decorates a method with an update of the mesh vertices.

        The method must have signature f(self, lmp, ntimestep, nlocal, tag, x,f \*args, \*\*kwargs) with
        x being the new vertex coordinates.
        """
        def wrap(self,  lmp, ntimestep, nlocal, tag, x,f,  *args, **kwargs):

            self.mesh.x = x[:self.n_vertices].reshape(self.mesh.x[:self.n_vertices].shape)
            #self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))
            return func(self, lmp, ntimestep, nlocal, tag, x,f, *args, **kwargs)
        wrap.__doc__  = func.__doc__
        wrap.__name__ = func.__name__
        return wrap

    @_update_mesh_one
    def callback_one(self, lmp, ntimestep, nlocal, tag, x, f):
        """!!!!!!!This function is used as callback to TRIMEM FROM LAMMPS!!!!!"""
        #print(tag)
        #tag_clear=[x-1 for x in tag if x <= self.n_vertices]
        f[:self.n_vertices]=-self.estore.gradient(self.mesh.trimesh)

        ## UNCOMMENT IF TRIMEM SHOULD GET THE ENERGY IN REALTIME
        #self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))

    @_update_mesh_one
    def callback_harm(self, lmp, ntimestep, nlocal, tag, x, f):
        """ SIMPLE HARMONIC FORCE TO TEST CALLBACK FUNCTIONALITY """
        f[:,0] = -(x[:,0]-2)
        f[:,1] = np.zeros_like(x[:,1])
        f[:, 2] = np.zeros_like(x[:, 2])

        #print(np.max(f))
        # self.lmp.fix_external_set_energy_global("ext", self.estore.energy(self.mesh.trimesh))


    #### for minimization
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


    #############################
    #    CALLBACK DURING .run() #
    ################################################################################
    # This functions performs statistics / output / etc. during the simulation run #
    ################################################################################
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
        i = sum(steps.values()) #py3.10: steps.total()

        if self.output_params.info and (i % self.output_params.info == 0):
            print("\n-- Energy-Evaluation-Step ", i)
            self.estore.print_info(self.mesh.trimesh)
        if self.output_params.thin and (i % self.output_params.thin == 0):
            self.output(i)
            #self.output.write_points_cells(self.mesh.x, self.mesh.f)

            #bonds_lmp = self.lmp.numpy.gather_bonds()[:, 1:3]
            #bonds_lmp = np.unique(bonds_lmp, axis=0)
            #bonds_lmp = (np.sort(bonds_lmp, axis=1))

            #with open('bonds_topo.xyz','a+') as f:
            #    for i in range(bonds_lmp.shape[0])
            #    f.write(f'{i}')

        if self.output_params.checkpoint_every and (i % self.output_params.checkpoint_every == 0):
            self.cpt_writer()

        self.estore.update_reference_properties()


        if self.output_params.energy_increment and (i % self.output_params.energy_increment==0):

            # MMB CHANGE -- PRINT ENERGIES
            self.ke_new=self.lmp.numpy.extract_compute("th_ke",LMP_STYLE_GLOBAL,LMP_TYPE_SCALAR)
            self.pe_new=self.lmp.numpy.extract_compute("th_pe", LMP_STYLE_GLOBAL, LMP_TYPE_SCALAR)
            
            # MMB compute volume and area of the mesh
            test_mesh = trimesh.Trimesh(vertices=self.mesh.x, faces=self.mesh.f)
            mesh_volume = test_mesh.volume
            mesh_area   = test_mesh.area
            with open(f'{self.output_params.output_prefix}_system.dat','a+') as f:
                f.write(f'{i} {self.estore.energy(self.mesh.trimesh)} {self.ke_new} {self.pe_new} {self.acceptance_rate} {mesh_volume} {mesh_area}\n')
            
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

                self.timer.performance_timestamps.pop(0)
                #{self.process.cpu_percent(interval=None): .4f}

    #####################################
    #   MINIMIZE HELFRICH HAMILTONIAN   #
    ###################################################################
    #       PRECONDITIONING USING STANDARD TRIMEM FUNCTIONALITY       #
    ###################################################################
    # See Trimem documentation for details                            #

    def minim(self):
        """Run (precursor) minimization.

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




    #########################################
    #         CHECKPOINT CREATION           #
    #####################################################################################
    #                PICKLE REDUCTION USED FOR CHECKPOINT CREATION!!!                   #
    #####################################################################################

    def __reduce__(self):
        return self.__class__,(self.initialize,
                self.mesh.x,
                self.mesh.f,
                self.lmp.numpy.extract_atom('v')[:self.n_vertices,:],
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
                self.algo_params.langevin_thermostat,
                self.algo_params.langevin_damp,
                self.algo_params.langevin_seed,
                self.algo_params.pure_MD,
                self.algo_params.switch_mode,
                self.algo_params.box,
                self.algo_params.additional_command,



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
                self.beads.n_types,
                self.beads.bead_interaction,
                self.beads.bead_interaction_params,
                self.lmp.numpy.extract_atom('x')[self.n_vertices:,:],
                self.lmp.numpy.extract_atom('v')[self.n_vertices:, :],
                self.beads.bead_sizes,
                self.beads.masses,
                self.beads.types,
                self.beads.self_interaction,
                self.beads.self_interaction_params
                               )


    # checkpoints using pickle
    def make_checkpoint_handle(self):
        return self.make_checkpoint

    def make_checkpoint(self, force_name=None):

        if not force_name:

            cptfname = pathlib.Path(self.output_params.output_prefix)
            cptfname = cptfname.name + self.output_params.output_flag + '.cpt'

            with open(cptfname, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)
            if self.output_params.output_flag == 'A':
                self.output_params.output_flag = 'B'
            else:
                self.output_params.output_flag = 'A'
        else:
            cptfname = pathlib.Path(force_name)
            with open(cptfname, 'wb') as f:
                pickle.dump(self, f, protocol=pickle.HIGHEST_PROTOCOL)

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
    

    # lammps
    def pair_cmds(self):
        def write_srp_table():
            """
            creates a python file implementing the repulsive part of the tether potential as surface repulsion readable by
            the lammps pair_style python
            uses the pair_style defined above to create a lookup table used as actual pair_style in the vertex-vertex interaction in Lammps
            is subject to 1-2, 1-3 or 1-4 neighbourhood exclusion of special bonds
            used to model the mesh topology
            """
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
            self.lmp.commands_string(dedent(f"""\
            pair_style python {self.eparams.repulse_params.lc1}
            pair_coeff * * trilmp_srp_pot.SRPTrimem C {'C '*self.beads.n_types}
            shell rm -f trimem_srp.table
            pair_write  1 1 2000 rsq 0.000001 {self.eparams.repulse_params.lc1} trimem_srp.table trimem_srp 1.0 1.0
            pair_style none 
            """))
        write_srp_table()

        
        pairs:list[tuple[str,float,str,str]]=[]
        overlay=False
        def add_pair(name:str,cutoff:float,args:str,modify_coeff_cmds:str):
            pairs.append((name,cutoff,args,modify_coeff_cmds))

        
        assert self.eparams.repulse_params.lc1
        # todo: performance : change to table bitmap style for improved performance
        # todo: performance : change to non-table style
        add_pair("table",self.eparams.repulse_params.lc1, "linear 2000",dedent(f"""
        pair_modify pair table special lj/coul 0.0 0.0 0.0
        pair_coeff 1 1 table trimem_srp.table trimem_srp
        """))
        
        if self.beads.n_beads:

            if self.beads.bead_interaction=='nonreciprocal':
                overlay=True
                # the bead_interaction_params should be set to 'nonreciprocal'
                # the parameters are handed as tuples (activity1,mobility1,activity2,mobility2,exponent,scale,k_harmonic,cut_mult)
                # k_harmonic determines height of bead_membrane harmonic barrier and k_harm_beads the bead-bead repulsion (if bead_self_interaction is set to True) (to do)
                if self.beads.bead_interaction_params[5]=='auto':
                    scale=(1+self.beads.bead_interaction_params[4])*0.5*(self.estore.eparams.bond_params.lc1 + self.beads.bead_sizes)*2**(-self.beads.bead_interaction_params[4])
                else:
                    scale=self.beads.bead_interaction_params[5]
                
                
                activity_1=self.beads.bead_interaction_params[0]
                mobility_1=self.beads.bead_interaction_params[1]
                activity_2=self.beads.bead_interaction_params[2]
                mobility_2=self.beads.bead_interaction_params[3]
            
                # soft-core (harmonic) repulsion
                k_harmonic=self.beads.bead_interaction_params[6]
                lc_harmonic_12=0.5*(self.estore.eparams.bond_params.lc1 + self.beads.bead_sizes)
                lc_harmonic_22=self.beads.bead_sizes

                add_pair("harmonic/cut","","",dedent(f"""\
                    pair_coeff * * harmonic/cut 0 0
                    pair_coeff 1 2 harmonic/cut {k_harmonic} {lc_harmonic_12}
                    pair_coeff 2 2 harmonic/cut {k_harmonic} {lc_harmonic_22}
                """))

                sigma12=float(0.5*(self.estore.eparams.bond_params.lc1 + self.beads.bead_sizes))
                cutoff_nonrec = float(sigma12*self.beads.bead_interaction_params[7])
                exponent=self.beads.bead_interaction_params[4]
                scale_nonrep=float(f"{scale:.4f}")
                add_pair("nonreciprocal", "",f"{cutoff_nonrec} {scale_nonrep} {exponent} {sigma12}",dedent(f"""
                    pair_coeff * * nonreciprocal 0 0 0 0 0
                    pair_coeff 1 2 nonreciprocal {activity_1} {activity_2} {mobility_1} {mobility_2} {cutoff_nonrec}
                """))

            elif self.beads.bead_interaction=='lj/cut':
                overlay=True
                bead_ljd = 0.5 * (self.estore.eparams.bond_params.lc1 + np.max(self.beads.bead_sizes))
                cutoff=4*bead_ljd
                cmds:list[str]=[]
                if self.beads.n_types==1:
                    cmds.append(dedent(f"""                    
                        pair_coeff * * {self.beads.bead_interaction} 0 0 0                   
                        pair_coeff 1 2 {self.beads.bead_interaction} {self.beads.bead_interaction_params[0]} {bead_ljd} {bead_ljd*self.beads.bead_interaction_params[1]}  
                        pair_coeff 2 2 {self.beads.bead_interaction} 0 0 0             
                        """))
                else:
                    for i in range(self.beads.n_types):
                        bead_ljd = 0.5 * (self.estore.eparams.bond_params.lc1 + self.beads.bead_sizes[i])
                        cmds.append(f'pair_coeff 1 {i+2} {self.beads.bead_interaction} {self.beads.bead_interaction_params[i][0]} {bead_ljd} {bead_ljd*self.beads.bead_interaction_params[i][1]}')
                        if self.beads.self_interaction:
                            for j in range(i,self.beads.n_types):
                                bead_ljd = 0.5 * (self.beads.bead_sizes[i] + self.beads.bead_sizes[i])
                                cmds.append(f'pair_coeff {i+2} {j + 2} {self.beads.bead_interaction} {self.beads.self_interaction_params[0]} {bead_ljd} {bead_ljd * self.beads.self_interaction_params[1]}')
                        else:
                            for j in range(i,self.beads.n_types):
                                cmds.append(f'pair_coeff {i + 2} {j + 2}  {self.beads.bead_interaction} 0 0 0')

                add_pair(str(self.beads.bead_interaction),float(cutoff),"", "\n".join(cmds))
            elif self.beads.bead_interaction=='custom':
                raise NotImplementedError()
                # self.lmp.commands_string(self.beads.bead_interaction_params)
        
        # todo: check if global cutoff is needed somewhere
        # global_cutoff=max((cutoff for (name,args,cutoff,cmds) in pairs),default=np.nan)

        def pair_style_cmd():
            def pair_style_2_style_cmd(name:str,cutoff:float,args:str):
                l=[name]
                if not name.startswith('table'):
                    l.append(str(cutoff))
                l.append(args)
                return ' '.join(l)
            pair_style='hybrid/overlay' if overlay else 'hybrid'
            l=['pair_style',pair_style]
            for (name,cutoff,args,*_) in pairs:
                l.append(pair_style_2_style_cmd(name,cutoff,args))
            return ' '.join(l)
        l=[]
        l.append(pair_style_cmd())
        for pair in pairs:
            l.append(pair[-1])
        return '\n'.join(l)

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
