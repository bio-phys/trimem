# TriLmp

**TriLmp** is a modified version of the **Trimem** python package for the Monte Carlo simulation of lipid
membranes according to the Helfrich theory [^Helfrich1973]. It allows for the direct use for MD simulations
in connection with LAMMPS via the python interface of the latter. Hereby the calculation of the surface repulsion
is dealt with by LAMMPS instead of Trimem.

[^Helfrich1973]: Helfrich, W. (1973) Elastic properties of lipid bilayers:
  Theory and possible experiments. Zeitschrift für Naturforschung C,
  28(11), 693-703

## Installation

We suggest installing TriLmp in a conda environment and then install trimem/trilmp as a editable install.
This allows you to alter trilmp functionality by directly tinkering with the src/trimem/trilmp.py file.
As trilmp is reliant on a lot of LAMMPS command-string which are not all parametrized via the TriLmp object,
an editable install should be a bit more flexible.


### Conda Environment

To set up a conda environment including some prerequisites use:

```bash
conda create -n Trienv
conda install -n Trienv scikit-build libgcc 
```
Make sure the evironment is activated throughout the rest of the setup!

TriLmp can be installed using pip:
Note that this folder will be the actual location of the modules with an editable install.
I.e. if you change something in the python code here, effects will be imediate.
In case you want to change something on the c++ side: run "python3 setup.py build" to compile and copy 
the libaries to the src/trimem folder as described below.

```bash
 git clone --recurse-submodules https://github.com/Saric-Group/trimem_sbeady.git
 git submodule update
 cd trimem_sbeady
 pip install -e .
```

Finally some shared libraries have to be copied manually from a build folder. 
The names depends on system and python verison!
-> all the .so files should be located in the src/trimem folder

```bash
cp _skbuild/linux-x86_64-3.11/cmake-build/core.cpython-39-x86_64-linux-gnu.so src/trimem/.
cp _skbuild/linux-x86_64-3.11/cmake-build/libtrimem.so src/trimem/.
cp _skbuild/linux-x86_64-3.11/cmake-build/Build/lib/* src/trimem/.
```

LAMMPs has to be installed using python shared libraries and some packages (can be extended).
If you already have lammps you might just have to reinstall it. In any case you will have to 
perform the python install using your Trienv environment.

To get lammps from git

```bash
git clone -b stable https://github.com/lammps/lammps.git lammps
```

Before proceeding further you have to place some files in the lammps/src folder to enable the nonreciprocal and nonreciprocal/omp pair_styles.
In the folder nonrec in the trimem_sbeady repositry you find pair_nonreciprocal.cpp/.h and pair_nonreciprocal_omp.cpp/.h .
The two files pair_nonreciprocal have to be placed in the lammps/src folder and the two pair_nonreciprocal_omp files in the lammps/src/OPENMP folder.
Now we can start installing the LAMMPS. First go to the lammps directory and create the build folder.

```bash
cd lammps                
mkdir build
cd build
```

Next we have to determine the python executable path. You can do that from the python console
using
```bash
python3                
import sys
print(sys.executable)
```
which should print something like 

```bash
/nfs/scistore15/saricgrp/Your_folder/.conda/envs/Trienv/bin/python3
```
and will be referred to as "python_path" in the next bash command. 
-D PKG_ASPHERE=yes is not stricly necessary bet here you can add whatever you have in mind for your simulations.

```bash
cmake -D BUILD_OMP=yes -D BUILD_SHARED_LIBS=yes -D PYTHON_EXECUTABLE="python_path" -D PKG_MOLECULE=yes -D PKG_PYTHON=yes -D PKG_OPENMP=yes -D PKG_EXTRA-PAIR=yes -D PKG_ASPHERE=yes ../cmake

cmake --build .
```

Finally to make LAMMPS accessible for python 

```bash
make install-python
```










### Dependencies


(Rest is CopyPaste from Trimem)
Trimem builds upon the generic mesh data structure
[OpenMesh](https://www.graphics.rwth-aachen.de/software/openmesh/), which
is included as a submodule that is pulled in upon `git clone` via the
`--recurse-submodules` flag.

For the efficient utilization of shared-memory parallelism, trimem makes
use of the [OpenMP](https://www.openmp.org/) application programming model
(`>= v4.5`) and modern `C++`. It thus requires relatively up-to-date
compilers (supporting at least `C++17`).

If not already available, the following python dependencies will be
automatically installed:

* numpy
* scipy
* h5py
* meshio
* omp_thread_count
* psutil

Documentation and tests further require:

* autograd
* trimesh
* sphinx
* sphinx-copybutton
* sphinxcontrib-programoutput

### Development installation

Unit-tests can be run with

```bash
pip install trimem/[tests]
pytest -v tests/
```

The documentation can be generated with `sphinx` by

```bash
pip install trimem/[docs]
cd doc; make html
```

## Usage

For an introduction to the usage of trimem please refer to the
[documentation](https://trimem.readthedocs.io/).

## Citation

If you use trimem for your scientific work, please consider the citation of
Siggel, M. et al[^Siggel2022].

[^Siggel2022]: Siggel, M. et al. (2022) TriMem: A Parallelized Hybrid Monte
  Carlo Software for Efficient Simulations of Lipid Membranes.
  J. Chem. Phys. (in press) (2022); https://doi.org/10.1063/5.0101118

## References

