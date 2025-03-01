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
conda create -y -p $deps_dir c-compiler cxx-compiler cmake ninja \
    tbb-devel tbb libboost-devel heyoka fmt spdlog 'python=3.12' \
    pybind11 clang clangxx clang-tools
source activate $deps_dir

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
    -DCMAKE_CXX_FLAGS_DEBUG="-g -Og" \
    -DCMAKE_CXX_CLANG_TIDY=`which clang-tidy` \
    -DCMAKE_C_CLANG_TIDY=`which clang-tidy` \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++

# Build. and install.
ninja -v

set +e
set +x
