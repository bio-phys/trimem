
#needed for initial state configuration
import trimesh
#trimem routines for estoer
from trimem.core import TriMesh
#np
import numpy as np
#for loading a checkpoint (instead of load checlpoint one could use pickle.load(filename.cpt)
import pickle

#the trilmp objects
from trimem.mc.trilmp import TriLmp, read_checkpoint, load_checkpoint

"""
example simulation for the use of trilmp to simulate nonreciprocal interactions of a spherical membrane with a number of external beads 
"""


##### CREATE INITIAL MESH-SPHERE AND SCALE TO BIGGER RADIUS (MAYBE NEEDED FOR HIGER DISCRETIZATIONS)
mesh = trimesh.creation.icosphere(5)
mesh.vertices=mesh.vertices*2



trilmp=TriLmp(mesh_points=mesh.vertices,  # input mesh
              mesh_faces=mesh.faces,
              initialize=True,            # use mesh to initialize mesh reference
              output_prefix='nonrec_test_trilmp',         # prefix for output filenames
              checkpoint_every=1000,     # interval of checkpoints (alternating pickles)
              thin=1,                     # write out
              num_steps=1000,             # number of steps in simulation (overwritten if trilmp.run(N=new_number)
              info=0,                     # output hmc/flip info every ith step
              performance_increment=10,   # output performace stats to prefix_performance.dat file
              energy_increment=1000,      # output energies to energies.dat file
              initial_temperature=1.0,    # initial temperature -> for HMC
              output_format='lammps_txt',  # choose different formats for 'lammps_txt' or 'h5_custom'


              ## NONRECIPTOCAL
              n_types=1,                 # different types of particles
              bead_pos=np.asarray([[-2.57,0,0],[+2.57,0,0],[0,+2.57,0],[0,-2.57,0],[0,0,+2.57],[0,0,-2.57]]),    # positions -> size also sets number
              bead_vel=np.asarray([[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0],[0,0,0]]),                            # initial velocities -> not always necessary
              bead_sizes=(1.0),                                                                                  # diameter of beads (tuple with dim n_types)
              bead_int='nonreciprocal',                                              # -> chosing nonreciprocal interactions
              bead_int_params=(14,     #activity1
                               -1,      #mobility1
                               80,      #activity2
                               -1,      #mobility2
                                7,       #exponent
                               'auto',  #scale                                       # if not auto, then numeric value is used for scale parameter!
                                5.0e6,    #k_harmonic
                               2.2),    #cut_mul_factor (cutoff = multiples of contact distance)
              bead_types=(2,2,2,2,2,2),                                              # types for beads (tuple/array dim n_beads )
              bead_masses=1,                                                         # mass per bead (single bead -> double, multiple beads -> tuple of doubles)
              kappa_b=30.0,                                                          # bending modulus

              thermal_velocities=False,  #no reset of vel at start of MD traj
              pure_MD=True,              #no metropolis rsetting of positions
              langevin_thermostat=True,  #use langevin thermostat -> BD sim
              langevin_damp=0.03,        #damping
              langevin_seed=1,           #seed for BD

              )



# create specific checkpoint to be used as e.g. as common starting point for different simulations
trilmp.make_checkpoint(force_name='test.cpt')

# load a checkpoint with a specific name -> otherwise load last checkpoint associated with output_prefix (see function definition in trilmp.py)
trilmp=load_checkpoint('test.cpt',alt='explicit')




# in order to change parameters associated with the helfrich hamiltonian, you have to access their storage in the estore object
trilmp.estore.eparams.kappa_c=0.0  # e.g. setting the harmonic potential for the mean curvature to 0 -> only local bending penalty for deformations
# to change stuff appearing on the LAMMPS side of things you need to create either a new trilmp object or, change LAMMPs specific parameters e.g. beads bodns etc,
# then make a checkpoint and reload -> the _init_() of the TriLmp class must be called to triggere a new lammps setup


# run simulation for trilmp.algo_params.num_steps steps , alternatively use trilmp.run(N=some number of steps)
trilmp.run()























