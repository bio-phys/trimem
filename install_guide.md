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

build lammps with openmp support, linked to python

`./lmp_build.sh`  (osx only, adapt as needed)

`./lmp_clean.sh` uninstalls it.

build shared libraries


`python setup.py build`

copy shared libraries to src 
`./copy_libs.sh`

install module in editable mode
`pip install -e .`

test
`python -c "from trimem import core"`

if re-compiling shared libraries, clean previous ones with:

`./clean.sh`
