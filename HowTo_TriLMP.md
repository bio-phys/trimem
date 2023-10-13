# TriLMP usage

**1. Resolution of the mesh/membrane (i.e., number of vertices in the mesh) [Before initializing trilmp object]** 

Controlled via the parameter ```r``` in ```trimesh.creation.icosphere(r, r_sphere = 1)```. 

You can simulate meshes with 42 beads (```r=1```), 162 beads (```r=2```), 642 beads (```r=3```), 2562 beads (```r=4```) or 10242 beads (```r=5```), for example. 

The command creates a sphere of radius ```r_sphere = 1``` by default. You can see several examples in the image below.

![DivisionMesh](https://github.com/Saric-Group/trimem_sbeady/assets/58335020/62779299-7cca-4667-820c-c7ff4c1cc3ad)

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

**3. Initialization of the trilmp object**

You can initialize the ```trilmp``` object by creating an instance of the ```TriLmp``` class:

```
my_trilmp_object = TriLmp(parameters)
```

TriLmp uses ```lj``` units for LAMMPS.


**4. Running a simulation**

To run a simulation, you need to use the ```run(N)``` method of the ```trilmp```. The parameter ```N``` controls the number of simulation steps, where a step consists of an MD run and an MC stage for bond flips (see details below). 

- The length of the MD run is controlled by the parameter ```traj_steps``` (introduced during ```trilmp``` initialization). The timestep used for time-integration during the MD part of the simulation is controlled by ```step_size```.
- The fraction of bonds in the membrane that we attempt to flip is controlled by ```flip_ratio``` (TriMEM is in charge of that).

A single simulation step can look two different ways, depending on the parameter ```switch_mode``` (options are 'alternating' and 'random')

1. If ```switch_mode=='alternating'```, a step consists of a MD simulation for ```traj_steps``` steps followed by an attempt to flip bonds in the mesh.
2. If ```switch_mode=='random'```, the program chooses randomly whether to run an MD simulation for ```traj_steps``` or to attempt to flip bonds in the mesh during a simulation step.

TriMEM was designed to perform hybrid MC simulations. This means that after the MD run, it will decide whether to accept or reject the resulting configuration based on a Metropolis criterion. Instead, if you want to run a pure MD simulation, you need to make sure that the flag ```pure_MD=True``` is on during the initialization of the ```trilmp``` object. This way, the program will accept all configurations that result from the MD run. Additionally, make sure that ```thermal_velocities=False``` so that the program keeps track of the velocities at the end of the MD run instead of resetting them according to the Maxwell-Boltzmann distribution.

**5. Opening the black box: A few notes on how the program works**

- The program relies on TriMEM to compute the Helfrich hamiltonian for the membrane. This hamiltonian is used to compute the forces in the MD section of the run (see callback functions in source code). Surface repulsion (i.e., preventing the membrane from self-intersecting) is included as a ```pair_style python```, and passed to LAMMPS as a table.  
