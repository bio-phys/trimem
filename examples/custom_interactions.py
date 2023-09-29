
import trimesh
from trimem.core import TriMesh
import numpy as np
import pickle

from trimem.mc.trilmp import TriLmp, load_checkpoint





"""
In this example we want to highlight the option to define arbitrary bead-membrane and
bead-bead interactions using the setting for bead_int='custom' .
The membrane-membrane surface repulsion, necessary to avoid self-intersecting membranes
is implemented as a pair_style table/omp, created using the parameters used to initilize the TriLmp class.
To stay consistent with these one has to add additional interactions using hybrid or hybrid/overlay pairstyle.
A minimal example can be found in the definition of the set_repulsion() method of TriLmp:

                            pair_style hybrid lj/cut/omp 0.1 table/omp linear 2000
                            pair_coeff 1 1 table/omp trimem_srp.table trimem_srp
                            pair_modify pair table/omp special lj/coul 0.0 0.0 0.0
                            pair_coeff 2 2 lj/cut/omp 0.0 0.0
                            pair_coeff 1 2 lj/cut/omp 0.0 0.0
                            
Which introduces additional to the surface repulsion via table/omp a lj/cut/omp pair style between beads and membranes.

As illustrative example for several fucionalities of TriLmp we want to simulate the following (unphysical) scenario:
Two beads of different size placed inside a spherical membrane of volume and area V and A. The reference values for the 
Voulume is set 0.2*V_init leading to a apprupt shrinking of the membrane while overall area is more or less conserved.
To make the dynamics not to aprupt we set the coefficient for the volume control kappa_v=1e4 (in contrast to 1e6 being the stiff default).
The bead interacts with the membrane with a combination of a lj for the core and a longer-ranged repulsive harmonic potential.
The two beads repell each other with combination of lj and harmonic as well but relatively strong

To use the right size of the membrane beads we use the default size given bi 0.75*l with l being the average 
distance in the initial configuration"""

# to get the avg distance we use a method from the Trimem shared libray
import trimesh
import trimem.core as m
from trimem.core import TriMesh
from trimem.mc.mesh import Mesh


mesh = trimesh.creation.icosphere(5)
mesh.vertices=mesh.vertices*2
a, l = m.avg_tri_props(Mesh(points=mesh.vertices, cells=mesh.faces).trimesh)

sigma12=0.5*(1.0+0.75*l)  # assuming a bead 1 of diameter 1
sigma13=0.5*(1.5+0.75*l)  # assuming a bead 2 of diameter 1.5
sigma23=0.5*(1.5+1.0)

"""
By using the parameter bead_int='custom' the bead_int_params will be used as a command string.
Note that the parameter additonal_command could also be used to overwrite previous commands in the LAMMPS initialisation.
So to set the interactions we have to create a string block like the following
"""

custom_interactions=f"""
pair_style hybrid/overlay lj/cut/omp 1.0 harmonic/cut/omp table/omp linear 2000
                            pair_coeff 1 1 table/omp trimem_srp.table trimem_srp
                            pair_modify pair table/omp special lj/coul 0.0 0.0 0.0
                            pair_coeff 2 2 lj/cut/omp 0.0 0.0
                            pair_coeff 3 3 lj/cut/omp 0.0 0.0
                            pair_coeff 1 2 lj/cut/omp 1.5 {sigma12} {2.2*sigma12}
                            pair_coeff 1 3 lj/cut/omp 1.5 {sigma13} {2.2*sigma13}
                            pair_coeff 2 3 lj/cut/omp 1.0 {sigma23} {2.2*sigma23}
                            pair_coeff 1 2 harmonic/cut/omp  10 {2.2*sigma12}
                            pair_coeff 1 3 harmonic/cut/omp  10 {2.2*sigma13}
                            
                            pair_coeff 1 1 harmonic/cut/omp  0.0 0.0
                            pair_coeff 2 2 harmonic/cut/omp  0.0 0.0
                            pair_coeff 3 3 harmonic/cut/omp  0.0 0.0
                            
                            pair_coeff 2 3 harmonic/cut/omp  500.0 {4*sigma23}
                            
"""


""" 
Now we only have to initialize the TriLmp Class accordingly and use the .run() method
"""




trilmp=TriLmp(mesh_points=mesh.vertices,  # input mesh
              mesh_faces=mesh.faces,
              initialize=True,  # use mesh to initialize mesh reference
              output_prefix='custom_trilmp',  # prefix for output filenames
              checkpoint_every=1000,  # interval of checkpoints (alternating pickles)
              thin=1,  # write out
              num_steps=5000,  # number of steps in simulation (overwritten if trilmp.run(N=new_number)
              info=0,  # output hmc/flip info every ith step
              performance_increment=10,  # output performace stats to prefix_performance.dat file
              energy_increment=1000,  # output energies to energies.dat file
              initial_temperature=1.0,  # initial temperature -> for HMC
              output_format='lammps_txt_folder',  # choose different formats for 'lammps_txt', 'lammps_txt_folder' or 'h5_custom'
              ## <- this setting creates dict /lmp_trj

              n_types=2,
              bead_pos=np.asarray([[-1.49,0,0],[+1.23,0,0]]),       # place bead inside membrane
              bead_vel=None,   # no intial velocity for beads
              bead_sizes=(1.0,1.5),          # diameter of bead
              bead_int='custom',         # use custom setting to overwrite existing settings
              bead_int_params=custom_interactions,
              bead_types=(2,3),  # <- bead type for single particle (membrane -> type 1, beads type 2,3,....)
              bead_masses=(1.0,1.0),
              kappa_v=1e4,                   # make volume constraint less stiff
              volume_frac=0.2,               # set referrence value for membrane volume to 0.2*V0
              kappa_c=0.0,                   # turn of constraint on mean curvature
              box=(-5,5,-5,5,-5,5),    # set box manually



              thermal_velocities=False,  # no reset of vel at start of MD traj
              pure_MD=True,  # no metropolis rsetting of positions
              langevin_thermostat=True,  # use langevin thermostat -> BD sim
              langevin_damp=0.01,  # damping
              langevin_seed=1,  # seed for BD
              additional_command=None   # here you could also add the custom_interactions
                                         # but this parameter is intedend for manipulating/overwriting other lammps params
              )

# create specific checkpoint to be used as e.g. as common starting point for different simulations
trilmp.make_checkpoint(force_name='test.cpt')

# load a checkpoint with a specific name -> otherwise load last checkpoint associated with output_prefix (see function definition in trilmp.py)
trilmp=load_checkpoint('test.cpt',alt='explicit')

trilmp.run()





















