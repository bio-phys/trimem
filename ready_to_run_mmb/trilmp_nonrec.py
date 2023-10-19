import trimesh
from trimem.core import TriMesh
import numpy as np
import pickle
from trimem.mc.trilmp import TriLmp
import os

# create directory to save data
os.system('mkdir -p data')

# initialization of the membrane mesh
mesh = trimesh.creation.icosphere(4)
sigma=1.0
desired_average_distance = 2**(1.0/6.0) * sigma
current_average_distance = np.mean(mesh.edges_unique_length)
scaling = desired_average_distance/current_average_distance
mesh.vertices *= scaling

# get coordinates to place a nanoparticle
sigma_nanop=10
sigma_tilde = 0.5*(sigma_nanop+sigma)
coord_bead = mesh.vertices[0]
rtemp = np.sqrt(coord_bead[0]**2 + coord_bead[1]**2 + coord_bead[2]**2)
buffering=1.05
x = coord_bead[0] + buffering*sigma_tilde*coord_bead[0]/rtemp
y = coord_bead[1] + buffering*sigma_tilde*coord_bead[1]/rtemp
z = coord_bead[2] + buffering*sigma_tilde*coord_bead[2]/rtemp
coord_nanop=np.asarray([[x, y, z]])

# mechanical properties of the membrane
kappa_b = 20.0
kappa_a = 1.0e6
kappa_v = 1.0e6
kappa_c = 0.0
kappa_t = 1.0e5
kappa_r = 1.0e3

# MD properties
step_size = 1e-4
traj_steps = 100
langevin_damp = 1.0
langevin_seed = 123

# MC/TRIMEM bond flipping properties
flip_ratio=0.15

# simulation step structure
switch_mode='alternating' # of 'random'
total_number_steps=20000

# ouput and printing
print_frequency = 100

# initialization of the trilmp object -- writing out all values for initialization
trilmp=TriLmp(initialize=True,                          # use mesh to initialize mesh reference
              mesh_points=mesh.vertices,                # input mesh vertices 
              mesh_faces=mesh.faces,                    # input of the mesh faces
              kappa_b=kappa_b,                          # MEMBRANE MECHANICS: bending modulus (kB T)
              kappa_a=kappa_a,                          # MEMBRANE MECHANICS: constraint on area change from target value (kB T)
              kappa_v=kappa_v,                          # MEMBRANE MECHANICS: constraint on volume change from target value (kB T)
              kappa_c=kappa_c,                          # MEMBRANE MECHANICS: constraint on area difference change (understand meaning) (kB T)
              kappa_t=kappa_t,                          # MEMBRANE MECHANICS: tethering potential to constrain edge length (kB T)
              kappa_r=kappa_r,                          # MEMBRANE MECHANICS: repulsive potential to prevent surface intersection (kB T)
              step_size=step_size,                      # FLUIDITY ---- MD PART SIMULATION: timestep of the simulation
              traj_steps=traj_steps,                    # FLUIDITY ---- MD PART SIMULATION: number of MD steps before bond flipping
              flip_ratio=flip_ratio,                    # MC PART SIMULATION: fraction of edges to flip?
              initial_temperature=1.0,                  # MD PART SIMULATION: temperature of the system
              langevin_damp=langevin_damp,              # MD PART SIMULATION: damping of the Langevin thermostat (as in LAMMPS)
              langevin_seed=langevin_seed,              # MD PART SIMULATION: seed for langevin dynamics
              pure_MD=True,                             # MD PART SIMULATION: accept every MD trajectory?
              switch_mode=switch_mode,                  # MD/MC PART SIMULATION: 'random' or 'alternating' flip-or-move
              box=(-50,50,-50,50,-50, 50),                     # MD PART SIMULATION: simulation box properties, periodic
              info=print_frequency,                     # OUTPUT: frequency output in shell
              thin=print_frequency,                     # OUTPUT: frequency trajectory output
              output_prefix='data/nr',                  # OUTPUT: prefix for output filenames
              restart_prefix='data/restart_nr',         # OUTPUT: name for checkpoint files
              checkpoint_every=print_frequency,         # OUTPUT: interval of checkpoints (alternating pickles)
              output_format='lammps_txt',               # OUTPUT: choose different formats for 'lammps_txt', 'lammps_txt_folder' or 'h5_custom'
              performance_increment=print_frequency,    # OUTPUT: output performace stats to prefix_performance.dat file
              energy_increment=print_frequency,         # OUTPUT: output energies to energies.dat file

              # include non-reciprocal interactions
              n_types=1,
              bead_pos=coord_nanop,
              bead_vel=([[0, 0, 0]]),
              bead_sizes=(sigma_nanop),
              bead_int='nonreciprocal',
              bead_int_params=(10,-1, 30, -1, 7, 1.0, 2000,2.5),
              bead_types=(2),
              bead_masses=1
              )

trilmp.run(total_number_steps)
