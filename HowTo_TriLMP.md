# TriLMP usage

**1. Resolution of the mesh/membrane (i.e., number of vertices in the mesh) [Before initializing trilmp object]** 

Controlled via the parameter ```r``` in ```trimesh.creation.icosphere(r, r_sphere = 1)```. 

You can simulate meshes with 42 beads (```r=1```), 162 beads (```r=2```), 642 beads (```r=3```), 2562 beads (```r=4```) or 10242 beads (```r=5```), for example. 

The command creates a sphere of radius ```r_sphere = 1``` by default. You can see several examples in the image below. The numbers correspond to the number of vertices in the mesh, while the number in parenthesis indicates the ```r``` value given to ```trimesh```.

![DivisionMesh](https://github.com/Saric-Group/trimem_sbeady/assets/58335020/c1f703f4-7071-4ad4-99f1-f2dc76404661)

***

**2. Rescaling the edge lengths and lengthscale definition [Before initializing trilmp object]** 

When you initially create your mesh, there will be an average edge length that depends both on ```r``` and ```r_sphere``` (see above). You can define the desired lengthscale of your system by rescaling the edge length. In the code below, we set the average edge length of the system to the position of the minimum in a Lennard-Jones potential (i.e., $r_{\min} = 2^{1/6}\sigma$).

```
sigma = 1
desired_average_distance = 2**(1.0/6.0)*sigma
current_average_distance = np.mean(mesh.edges_unique_length)
scaling = desired_average_distance/current_average_distance
mesh.vertices *= scaling
```

When perfoming this type of scaling keep in mind that different ```r``` values will give you spheres with different radii: the larger ```r```, the larger the radius of the sphere for the same average edge length.

***

**3. Initialization of the trilmp object**

You can initialize the ```trilmp``` object by creating an instance of the ```TriLmp``` class:

```
my_trilmp_object = TriLmp(parameters)
```

TriLmp uses ```lj``` units for LAMMPS.

***

**4. Running a simulation**

To run a simulation, you need to use the ```run(N)``` method of the ```trilmp``` object (i.e., add ```my_trilmp_object.run(N)``` once the ```trilmp``` object has been created). The parameter ```N``` controls the number of simulation steps, where a step consists of an MD run and an MC stage for bond flips (see details below). Please note than an MD run may consist of multiple steps itself.

- The length of the MD run is controlled by the parameter ```traj_steps``` (introduced during ```trilmp``` initialization). The timestep used for time-integration during the MD part of the simulation is controlled by ```step_size```.
- The fraction of bonds in the membrane that we attempt to flip is controlled by ```flip_ratio``` (TriMEM is in charge of that).

A single simulation step can look two different ways, depending on the parameter ```switch_mode``` (options are 'alternating' and 'random')

1. If ```switch_mode=='alternating'```, a step consists of a MD simulation for ```traj_steps``` steps followed by an attempt to flip bonds in the mesh.
2. If ```switch_mode=='random'```, the program chooses randomly whether to run an MD simulation for ```traj_steps``` or to attempt to flip bonds in the mesh during a simulation step.

TriMEM was designed to perform hybrid MC simulations. This means that after the MD run, it will decide whether to accept or reject the resulting configuration based on a Metropolis criterion. Instead, if you want to run a pure MD simulation, you need to make sure that the flag ```pure_MD=True``` is on during the initialization of the ```trilmp``` object. This way, the program will accept all configurations that result from the MD run. Additionally, make sure that ```thermal_velocities=False``` so that the program keeps track of the velocities at the end of the MD run instead of resetting them according to the Maxwell-Boltzmann distribution.

***

**5. Sampling data + printing out energies**

The program prints outputs at a user defined frequency.

- Energy output (printed with frequency ```energy_increment``` steps) is stored in ```output_prefix```_system.dat. The columns in the file correspond to (1) total number of simulation steps (how many MD stages + how many MC stages; note that an MD stage contains ```traj_steps``` steps), (2) total Trimem hamiltonian, (3) LAMMPS ```compute ke```, (4) LAMMPS ```compute pe```, (5) volume of the membrane, (6) area of the membrane, (7) bending energy/k_B (MUST multiply by bending modulus to get actual bending energy, see eq. 3 in Siggel et al., JChemPhys 2022), (8) tethering energy and (8) curvature (see eq. 6 in Siggel et al., JChemPhys 2022).

***

**6. Opening the black box: A few notes on how the program works**

- The program relies on TriMEM to compute the Helfrich hamiltonian for the membrane. This hamiltonian is used to compute the forces in the MD section of the run (see callback functions in source code). Surface repulsion (i.e., preventing the membrane from self-intersecting) is included as a ```pair_style python```, and passed to LAMMPS as a table.  


***

# Validation and checks of the code

**1. Monitoring energy of the system**

We can monitor the value of the Trimem Hamiltonian (see Trimem documentation), the total kinetic energy and the acceptance rate for the bond flips by plotting the data in ```output_prefix```_system.dat (double check values for acceptance rate).  The figure below shows data for a membrane with ```r=4```, ```flip_ratio``` of 0.1 and MD trajectories with ```traj_steps=100``` steps. Note how this data also shows that the system is correctly equilibrating to the prescribed temperature, as $k_B T = 2/3 KE/N = 1.014$ (set temperature ```temperature = 1```).

![monitor_FR_0 1_TS_100_R_4](https://github.com/Saric-Group/trimem_sbeady/assets/58335020/1f697af6-99b4-4c53-98e0-46b6389e154c)

The total energy of the membrane depends on the mesh resolution (the number of vertices and number of edges), as shown in the figure below. It also depends on its mechanical properties (compare the data for floppy, where ```k_v = k_a = 1e5```, with rigid, where ```k_v = k_a = 1e6```). Additionally, it seems to depend on the length of the MD stage in the program for short MD simulations. The energy of the membrane does not depend on the ```flip_ratio``` parameter (all three curves for ```flip_ratio = 0.1, 0.2 and 0.3``` are practically identical).

![summary_energy](https://github.com/Saric-Group/trimem_sbeady/assets/58335020/fa98417f-2931-4cb3-b478-99d5a8924f90)


Simulations with short MD stages also seem to give slightly lower temperature than expected, as shown in the figure below (data shown corresponds to 'floppy' and 'rigid' membranes (see above) with ```flip_ratio = 0.1, 0.2 and 0.3```). The MD stages are simulated using the ```fix langevin``` from LAMMPS; we chose ```t_damp = 1```.

![summary_ke](https://github.com/Saric-Group/trimem_sbeady/assets/58335020/06bc54e5-e4ea-4aa1-a275-07a5e8212875)

**2. Performance evaluation**

The figure below shows the average time it takes to perform one simulation step (MD stage + MC flips) as a function of the length of the MD stage for two mesh sizes (```r=4``` (2562 beads) and ```r=5``` (10242 beads)), and different flip rates ```f```.

![performance_evaluation](https://github.com/Saric-Group/trimem_sbeady/assets/58335020/ec07ab88-ff3b-4539-a429-7153df0f2732)

