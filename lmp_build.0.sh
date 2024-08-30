#!/bin/bash
export CONDA_BASE="$(conda info --base)"
export CONDA_NEXT="$CONDA_BASE/envs/trimem"
./lmp_build.sh &> >(translate_env)