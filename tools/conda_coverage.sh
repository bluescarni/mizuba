#!/usr/bin/env bash

# Echo each command
set -x

# Exit on error.
set -e

# Core deps.
sudo apt-get install wget

# Install conda+deps.
wget https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-${CONDA_INSTALLER_ARCH}.sh -O miniconda.sh
export deps_dir=$HOME/local
export PATH="$HOME/miniconda/bin:$PATH"
bash miniconda.sh -b -p $HOME/miniconda
conda create -y -p $deps_dir c-compiler cxx-compiler cmake ninja \
    tbb-devel tbb libboost-devel heyoka fmt spdlog 'python=3.12' numpy \
    pybind11 astropy heyoka.py lcov requests polars
source activate $deps_dir

# Workaround: install sgp4 with pip
# because the conda package for sgp4 on aarch64
# seemingly does not ship with OMM support.
pip install sgp4

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
    -DCMAKE_CXX_FLAGS="--coverage" \
    -DMIZUBA_INSTALL_PATH="`pwd`/install" \
    -DCMAKE_CXX_FLAGS_DEBUG="-g -Og"

# Build and install.
ninja -v install

# Run the tests.
cd install
python -c "from mizuba.test import run_test_suite; run_test_suite()"

# Create lcov report
cd ..
lcov --capture --directory . --output-file coverage.info

set +e
set +x
