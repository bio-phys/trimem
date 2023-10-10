download trilmp from git

```
git clone [repo]
git submodule init
```


create a conda environment

`conda env create -f environment.yml`
`conda activate trilmp`

or

```
git clone --recurse-submodules [repo]
git submodule update
```

`cd trilmp_sbeady`

install module in editable mode
`pip install -e .`

build libraries
`python setup.py build`

copy shared libraries to src 
`. copy_libs.sh`

build lammps with openmp support, linked to python, and with nonrec interactions
`cp -r ./nonrec lammps/src/`
`lmp_build.sh`  (osx only, adapt as needed)

`lmp_clean.sh` uninstalls it.
