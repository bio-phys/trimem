
import trimesh
from trimem.core import TriMesh
import numpy as np
import pickle

from trimem.mc.trilmp import TriLmp

"""
This simulation displays the use of the lennard jones potential incl. self-interacting beads
"""


mesh = trimesh.creation.icosphere(5)
mesh.vertices=mesh.vertices*2



trilmp=TriLmp(mesh_points=mesh.vertices,  # input mesh
              mesh_faces=mesh.faces,
              initialize=True,            # use mesh to initialize mesh reference
              output_prefix='lj_test_trilmp',         # prefix for output filenames
              checkpoint_every=1000,     # interval of checkpoints (alternating pickles)
              thin=1,                     # write out
              num_steps=1000,             # number of steps in simulation (overwritten if trilmp.run(N=new_number)
              info=0,                     # output hmc/flip info every ith step
              performance_increment=10,   # output performace stats to prefix_performance.dat file
              energy_increment=1000,      # output energies to energies.dat file
              initial_temperature=1.0,    # initial temperature -> for HMC
              output_format='lammps_txt_folder',  # choose different formats for 'lammps_txt', 'lammps_txt_folder' or 'h5_custom'
                                        ## <- this setting creates dict /lmp_trj to which

              # n_types=3,                                   ## <- use three different bead types
              # bead_pos=np.asarray([[-2.51,0,0],[2.51,0,0],[0,-2.26,0],[0,2.26,0],[0,0,-2.13],[0,0,2.13]]), ## <- 6 postions
              # bead_vel=None,                                ## <- No initial velocity (default)
              # bead_int='lj/cut/omp',                        ## <- this ist the default option
              # bead_sizes=(1.0,0.5,0.25),                    ## <- different diameters
              # bead_int_params=((12,2.2),(8,2.2),(4,2.2)),    ##  <- three different lennard jones potential membrane (type 1) - beads (type 2 - type 4)

              # bead_types=(2,2,3,3,4,4),                      ##  <- two beads of each type
              # bead_masses=(1.2,1.1,1.0),                     ##  <- each type has different mass
              # self_interaction=True,                        ##  <- beads are interacting with each other via lj
              # self_interaction_params=(0.1,1.5),             ##  <- currently only one parameter is supported -> scaling lj with (epsilon,factor*sigma_ij) -> cutoff at 1.5 contact distance
              kappa_c=0.0,    ## turn off harmonic pot for mean curvature



              ### These are the default Parameters but
              thermal_velocities=False,  #no reset of vel at start of MD traj
              pure_MD=True,              #no metropolis rsetting of positions
              langevin_thermostat=True,  #use langevin thermostat -> BD sim
              langevin_damp=0.03,        #damping
              langevin_seed=1            #seed for BD
    )

trilmp.run()





















