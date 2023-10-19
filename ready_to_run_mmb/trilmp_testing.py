import trimesh
from trimem.core import TriMesh
import numpy as np
import pickle
from trimem.mc.trilmp import TriLmp
import os
 
# create directory to save data of interest
os.system('mkdir -p data')
 
# initialization of the membrane mesh
mesh = trimesh.creation.icosphere(4)
sigma=1.0
desired_average_distance = 2**(1.0/6.0) * sigma
current_average_distance = np.mean(mesh.edges_unique_length)
scaling = desired_average_distance/current_average_distance
mesh.vertices *= scaling
 
# mechanical properties of the membrane
kappa_b = 20.0
kappa_a = 1.0e5
kappa_v = 1.0e5
kappa_c = 0.0
kappa_t = 1.0e3
kappa_r = 1.0e3
 
# MD properties
step_size = 0.005
traj_steps = 50
langevin_damp = 1.0
langevin_seed = 123
 
# MC/TRIMEM bond flipping properties
flip_ratio=0.6
 
# simulation step structure
switch_mode='alternating'
total_sim_time=10000
total_number_steps=int(total_sim_time/(step_size*traj_steps))
 
# ouput and printing
discret_snapshots=10
print_frequency = int(discret_snapshots/(step_size*traj_steps))
 
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
              box=(-50,50,-50,50,-50, 50),              # MD PART SIMULATION: simulation box properties, periodic
              info=print_frequency,                     # OUTPUT: frequency output in shell
              thin=print_frequency,                     # OUTPUT: frequency trajectory output
              output_prefix='data/calibration',         # OUTPUT: prefix for output filenames
              restart_prefix='data/restart_calibraton', # OUTPUT: name for checkpoint files
              checkpoint_every=print_frequency,         # OUTPUT: interval of checkpoints (alternating pickles)
              output_format='lammps_txt',               # OUTPUT: choose different formats for 'lammps_txt', 'lammps_txt_folder' or 'h5_custom'
              output_counter=0,                         # OUTPUT: initialize trajectory number in writer class
              performance_increment=print_frequency,    # OUTPUT: output performace stats to prefix_performance.dat file
              energy_increment=print_frequency,         # OUTPUT: output energies to energies.dat file
              )
 
trilmp.run(total_number_steps)
 
