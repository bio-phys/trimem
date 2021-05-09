# Playing around with OpenMesh and the helfrich bending energy.

## Installation

The installation picks the compilers set in the variables $CC$ and $CXX$. It
falls back to a system's gcc installation if existent. On the mpcdf clusters
this is not recommended and it is better to set the compiler explicitly.

```bash
git clone --recurse-submodules https://gitlab.mpcdf.mpg.de/sebak/om-helfrich.git`
pip install ./om-helfrich
```
## Usage on the mpcdf clusters

No particular adaptions are necessary when using the intel compilers. However,
when using the gnu compilers (and thus libgomp) there are issues with mixing
with intel's omp implementation used by anaconda-\>numpy-\>mkl. This can be
resolved by setting MKL\_THREADING\_LAYER=GNU in the slurm script.
