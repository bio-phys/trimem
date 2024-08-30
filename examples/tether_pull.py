
import trimesh
from trimem.core import TriMesh
import numpy as np
import pickle

from trimem.mc.trilmp import TriLmp







mesh = trimesh.creation.icosphere(5)
mesh.vertices=mesh.vertices*2


trilmp=TriLmp(mesh_points=mesh.vertices,  # input mesh
              mesh_faces=mesh.faces,
              initialize=True,  # use mesh to initialize mesh reference
              output_prefix='tether_test_trilmp',  # prefix for output filenames
              checkpoint_every=1000,  # interval of checkpoints (alternating pickles)
              thin=1,  # write out
              num_steps=1000,  # number of steps in simulation (overwritten if trilmp.run(N=new_number)
              info=0,  # output hmc/flip info every ith step
              performance_increment=10,  # output performace stats to prefix_performance.dat file
              energy_increment=1000,  # output energies to energies.dat file
              initial_temperature=1.0,  # initial temperature -> for HMC
              output_format='lammps_txt_folder',  # choose different formats for 'lammps_txt', 'lammps_txt_folder' or 'h5_custom'
              ## <- this setting creates dict /lmp_trj to which

              n_types=1,
              bead_pos=np.asarray([[-2.1,0,0]]),
              bead_vel=np.asarray([[-0.2,0,0]]),
              bead_sizes=(0.1),
              bead_int='tether',  # every bead is linked to it's closest membrane vertex
              bead_int_params=(1e6,0.1),  # <- harmonic/omp args -> (K_harm, r_0)
              bead_types=(2),  # <- bead type for single particle
              bead_masses=1e7,  # <- high mass of particle automatically scales langevin thermostat
              kappa_b=30.0,
              kappa_c=0.0,

              thermal_velocities=False,  # no reset of vel at start of MD traj
              pure_MD=True,  # no metropolis rsetting of positions
              langevin_thermostat=True,  # use langevin thermostat -> BD sim
              langevin_damp=0.03,  # damping
              langevin_seed=1  # seed for BD
              )



trilmp.run()





















