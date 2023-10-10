#!/bin/bash
pip uninstall lammps
PREFIX="$CONDA_PREFIX"
SHLIB_EXT='.dylib'

rm -rdf "${PREFIX}/include/lammps"
rm "${PREFIX}/lib/liblammps*${SHLIB_EXT}"
rm "${PREFIX}/bin/lmp_mpi"