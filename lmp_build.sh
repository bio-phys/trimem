#!/usr/bin/env bash
prefix="$CONDA_PREFIX"
conda deactivate
export PREFIX="$prefix"
export CONDA_BUILD=1
conda activate "$PREFIX"
set -euo pipefail

if [ "$(uname)" == "Darwin" ]; then
	export SHLIB_EXT='.dylib'
	# export target_platform="osx-arm64"
	CMAKE_ARGS+=" -DCMAKE_FIND_FRAMEWORK=NEVER -DCMAKE_FIND_APPBUNDLE=NEVER"
	if [ -z "$HOMEBREW_PREFIX" ]; then
		CMAKE_ARGS+=" -DCMAKE_IGNORE_PATH=${HOMEBREW_PREFIX}/bin;${HOMEBREW_PREFIX}/lib;/usr/bin"
	fi
	CMAKE_ARGS+=" -DCMAKE_IGNORE_PATH_PREFIX=${HOMEBREW_PREFIX}"
	CXXFLAGS=" -Wno-deprecated-declarations $CXXFLAGS"
	CXXFLAGS="${CXXFLAGS} -DTARGET_OS_OSX=1"
else
	export SHLIB_EXT='.so'
fi


cd lammps || exit
rm -rdf build
mkdir build
cd build || exit



args=""
# args+=" -D PKG_ASPHERE=ON"
# args+=" -D PKG_BODY=ON"
# args+=" -D PKG_BROWNIAN=ON"
# args+=" -D PKG_CORESHELL=ON"
args+=" -D PKG_DIPOLE=ON"
args+=" -D PKG_EXTRA-COMPUTE=ON"
args+=" -D PKG_EXTRA-DUMP=ON"
args+=" -D PKG_EXTRA-FIX=ON"
args+=" -D PKG_EXTRA-MOLECULE=ON"
args+=" -D PKG_EXTRA-PAIR=ON"
# args+=" -D PKG_GRANULAR=ON"
# args+=" -D PKG_H5MD=ON"
args+=" -D PKG_MISC=ON"
args+=" -D PKG_MOLECULE=ON"
# args+=" -D PKG_NETCDF=ON"
args+=" -D PKG_OPT=ON"
args+=" -D PKG_PLUGIN=ON"
# args+=" -D PKG_REPLICA=ON"
args+=" -D PKG_RIGID=ON"
# args+=" -D WITH_GZIP=ON"


# Plugins - n2p2 and latte
# if [[ -z "$MACOSX_DEPLOYMENT_TARGET" ]]; then
#   export LDFLAGS="-L$PREFIX/lib -lcblas -lblas -llapack -fopenmp $LDFLAGS"
# else
# #   export LDFLAGS="-fopenmp $LDFLAGS"
#   CXXFLAGS="${CXXFLAGS} -DTARGET_OS_OSX=1"
# fi
# CXXFLAGS="${CXXFLAGS} -DTARGET_OS_OSX=1"

# pypy does not support LAMMPS internal Python
PYTHON_IMPL=$(python -c "import platform; print(platform.python_implementation())")
if [ "$PYTHON_IMPL" != "PyPy" ]; then
args+=" -D PKG_PYTHON=ON -D Python_ROOT_DIR=${PREFIX} -D Python_FIND_STRATEGY=LOCATION"
fi

# Serial
# mkdir build_serial
# cd build_serial
# cmake -D BUILD_MPI=OFF -D BUILD_OMP=ON -D PKG_MPIIO=OFF $args ${CMAKE_ARGS} ../cmake
# make -j${NUM_CPUS}
# cp lmp $PREFIX/bin/lmp_serial
# cd ..

# Parallel and library
# export LDFLAGS="-L$PREFIX/lib $LDFLAGS"

# mkdir build_mpi
# cd build_mpi

args+=" -D WITH_JPEG=OFF"
args+=" -D WITH_PNG=OFF"
args+=" -D WITH_FFMPEG=OFF"
args+=" -D BUILD_OMP=ON"
args+=" -D PKG_OPENMP=ON"
args+=" -D BUILD_MPI=ON"
args+=" -D LAMMPS_EXCEPTIONS=yes"
args+=" -D BUILD_LIB=ON"
args+=" -D LAMMPS_INSTALL_RPATH=ON"
args+=" -D BUILD_SHARED_LIBS=ON"

args+=" ${CMAKE_ARGS}"

# configure
cmake --install-prefix "${PREFIX}" ${args} ../cmake

# build
cmake --build . -j "$(nproc)"

# install
# library
cp lmp "$PREFIX/bin/lmp_mpi"
cp -a liblammps*${SHLIB_EXT}* "${PREFIX}"/lib/
# header
mkdir -p "$PREFIX/include/lammps"
cp ../src/library.h "$PREFIX/include/lammps"

# python module
cd ../python
python -m pip install . --no-deps -vv
