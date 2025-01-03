#!/usr/bin/env bash

# Echo each command.
set -x

# Exit on error.
set -e

# Report on the environrnt variables used for this build.
echo "MIZUBA_BUILD_TYPE: ${MIZUBA_BUILD_TYPE}"
echo "GITHUB_REF: ${GITHUB_REF}"
echo "GITHUB_WORKSPACE: ${GITHUB_WORKSPACE}"
# No idea why but this following line seems to be necessary (added: 18/01/2023)
git config --global --add safe.directory ${GITHUB_WORKSPACE}
BRANCH_NAME=`git rev-parse --abbrev-ref HEAD`
echo "BRANCH_NAME: ${BRANCH_NAME}"

# Detect the Python version.
if [[ ${MIZUBA_BUILD_TYPE} == *38* ]]; then
	PYTHON_DIR="cp38-cp38"
elif [[ ${MIZUBA_BUILD_TYPE} == *39* ]]; then
	PYTHON_DIR="cp39-cp39"
elif [[ ${MIZUBA_BUILD_TYPE} == *310* ]]; then
	PYTHON_DIR="cp310-cp310"
elif [[ ${MIZUBA_BUILD_TYPE} == *311* ]]; then
	PYTHON_DIR="cp311-cp311"
elif [[ ${MIZUBA_BUILD_TYPE} == *312* ]]; then
	PYTHON_DIR="cp312-cp312"
elif [[ ${MIZUBA_BUILD_TYPE} == *313* ]]; then
	PYTHON_DIR="cp313-cp313"
else
	echo "Invalid build type: ${MIZUBA_BUILD_TYPE}"
	exit 1
fi

# Report the inferred directory where python is found.
echo "PYTHON_DIR: ${PYTHON_DIR}"

# The version of the heyoka C++ library to be used.
export HEYOKA_VERSION="7.2.0"

# Check if this is a release build.
if [[ "${GITHUB_REF}" == "refs/tags/v"* ]]; then
    echo "Tag build detected"
	export MIZUBA_RELEASE_BUILD="yes"
else
	echo "Non-tag build detected"
fi

# In the manylinux image in dockerhub the working directory is /root/install, we will install heyoka there.
cd /root/install

# Install heyoka.
curl -L -o heyoka.tar.gz https://github.com/bluescarni/heyoka/archive/refs/tags/v${HEYOKA_VERSION}.tar.gz
tar xzf heyoka.tar.gz
cd heyoka-${HEYOKA_VERSION}

mkdir build
cd build
cmake -DHEYOKA_WITH_MPPP=yes \
    -DHEYOKA_WITH_SLEEF=yes \
    -DHEYOKA_ENABLE_IPO=ON \
    -DHEYOKA_FORCE_STATIC_LLVM=yes \
    -DHEYOKA_HIDE_LLVM_SYMBOLS=yes \
    -DCMAKE_BUILD_TYPE=Release ../;
make -j4 install

# Build the mizuba wheel.
cd ${GITHUB_WORKSPACE}
/opt/python/${PYTHON_DIR}/bin/pip wheel . -v
export WHEEL_FILENAME=`ls mizuba*.whl`
echo "WHEEL_FILENAME: ${WHEEL_FILENAME}"
# Repair it.
# NOTE: this is temporary because some libraries in the docker
# image are installed in lib64 rather than lib and they are
# not picked up properly by the linker.
export LD_LIBRARY_PATH="/usr/local/lib64:/usr/local/lib"
auditwheel repair ./${WHEEL_FILENAME} -w ./repaired_wheel
export REPAIRED_WHEEL_FILENAME=`basename \`ls ./repaired_wheel/*.whl\``
echo "REPAIRED_WHEEL_FILENAME: ${REPAIRED_WHEEL_FILENAME}"
# Try to install it and run the tests.
unset LD_LIBRARY_PATH
cd /
/opt/python/${PYTHON_DIR}/bin/pip install ${GITHUB_WORKSPACE}/repaired_wheel/${REPAIRED_WHEEL_FILENAME}[sgp4,heyoka]
/opt/python/${PYTHON_DIR}/bin/python -c "import mizuba; mizuba.test.run_test_suite();"

# Upload to PyPI.
if [[ "${MIZUBA_RELEASE_BUILD}" == "yes" ]]; then
	/opt/python/${PYTHON_DIR}/bin/pip install twine
	/opt/python/${PYTHON_DIR}/bin/twine upload -u __token__ ${GITHUB_WORKSPACE}/repaired_wheel/${REPAIRED_WHEEL_FILENAME}
fi

set +e
set +x
