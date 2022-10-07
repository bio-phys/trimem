# Trimem

**Trimem** is a python package for the Monte Carlo simulation of lipid
membranes according to the Helfrich theory [^Helfrich1973].

[^Helfrich1973]: Helfrich, W. (1973) Elastic properties of lipid bilayers:
  Theory and possible experiments. Zeitschrift für Naturforschung C,
  28(11), 693-703

## Installation

Trimem can be installed using pip:

```bash
 git clone --recurse-submodules https://github.com/bio-phys/trimem.git
 pip install trimem/
```

We suggest the installation using the `--user` flag to `pip`. Alternatively,
we recommend to consider the usage of virtual environments to isolate the
installation of trimem, see, e.g., [here](https://packaging.python.org/en/latest/guides/installing-using-pip-and-virtual-environments/#creating-a-virtual-environment).

### Dependencies

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

