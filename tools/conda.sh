#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${CONDA_INSTALLER_ARCH}.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda create -y -p $deps_dir cmake ninja \
    tbb-devel tbb libboost-devel heyoka fmt spdlog 'python=3.12' numpy \
    pybind11 pandas astropy heyoka.py requests polars
source activate $deps_dir

# NOTE: not sure what is going on with conda OSX, but somehow
# at this time installing c/cxx-compiler fetches a version
# of clang(xx) earlier than the one used to produce the binary
# packages.
if [[ "${CONDA_INSTALLER_ARCH}" == "MacOSX"* ]]; then
    conda install -y 'clang=18.*' 'clangxx=18.*'
    # For some reason, we also really need to do this,
    # or the system compiler is being picked up.
    export CC=clang
    export CXX=clang++
else
    conda install -y c-compiler cxx-compiler
fi

# Workaround: install sgp4 and skyfield with pip
# because the conda package for sgp4 on aarch64
# seemingly does not ship with OMM support.
pip install skyfield sgp4

# Create the build dir and cd into it.
mkdir build
cd build

# Clear the compilation flags set up by conda.
unset CXXFLAGS
unset CFLAGS

# Configure.
cmake ../ -G Ninja \
    -DCMAKE_PREFIX_PATH=$deps_dir \
    -DCMAKE_BUILD_TYPE=Debug \
    -DCMAKE_CXX_FLAGS_DEBUG="-g -Og"

# Build and install.
ninja -v install

# Run the tests.
cd
python -c "from mizuba.test import run_test_suite; run_test_suite()"

set +e
set +x
