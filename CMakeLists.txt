cmake_minimum_required(VERSION 3.18.0)

# Set default build type to "Release".
# NOTE: this should be done before the project command since the latter can set
# CMAKE_BUILD_TYPE itself (it does so for nmake).
if(NOT CMAKE_BUILD_TYPE)
	set(CMAKE_BUILD_TYPE Release CACHE STRING
		"Choose the type of build, options are: None Debug Release RelWithDebInfo MinSizeRel."
	FORCE)
endif()

project(mizuba VERSION 0.0.1 LANGUAGES CXX C)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/yacma")

message(STATUS "System name: ${CMAKE_SYSTEM_NAME}")
message(STATUS "mizuba version: ${mizuba_VERSION}")

option(MIZUBA_ENABLE_IPO "Enable IPO (requires compiler support)." OFF)
mark_as_advanced(MIZUBA_ENABLE_IPO)

# Run the YACMA compiler setup.
include(YACMACompilerLinkerSettings)

# Assemble the flags.
set(MIZUBA_CXX_FLAGS_DEBUG ${YACMA_CXX_FLAGS} ${YACMA_CXX_FLAGS_DEBUG})
set(MIZUBA_CXX_FLAGS_RELEASE ${YACMA_CXX_FLAGS})

if(YACMA_COMPILER_IS_MSVC)
    include(CheckCXXCompilerFlag)
    # Disable the idiotic minmax macros on MSVC (both cl and clang-cl).
    # Also, enable the bigobj flag and the WIN32_LEAN_AND_MEAN definitions:
    # https://stackoverflow.com/questions/11040133/what-does-defining-win32-lean-and-mean-exclude-exactly
    list(APPEND MIZUBA_CXX_FLAGS_DEBUG "-DNOMINMAX" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
    list(APPEND MIZUBA_CXX_FLAGS_RELEASE "-DNOMINMAX" "/bigobj" "-DWIN32_LEAN_AND_MEAN")
    # Enable strict conformance mode, if supported.
    set(CMAKE_REQUIRED_QUIET TRUE)
    check_cxx_compiler_flag("/permissive-" _MIZUBA_MSVC_SUPPORTS_STRICT_CONFORMANCE)
    unset(CMAKE_REQUIRED_QUIET)
    if(_MIZUBA_MSVC_SUPPORTS_STRICT_CONFORMANCE)
        message(STATUS "The '/permissive-' flag is supported, enabling it.")
        list(APPEND MIZUBA_CXX_FLAGS_DEBUG "/permissive-")
        list(APPEND MIZUBA_CXX_FLAGS_RELEASE "/permissive-")
    endif()
    unset(_MIZUBA_MSVC_SUPPORTS_STRICT_CONFORMANCE)
    if(YACMA_COMPILER_IS_CLANGXX)
        # clang-cl emits various warnings from GMP/MPFR, let's just silence them.
        # NOTE: at one point in the recent past, MSVC added an options similar to GCC's isystem:
        # https://blogs.msdn.microsoft.com/vcblog/2017/12/13/broken-warnings-theory/
        # We probably just need to wait for this to be picked up by CMake/clang-cl. Let's
        # revisit the issue in the future.
        list(APPEND _MIZUBA_CLANG_CL_DISABLED_WARNINGS
            "-Wno-unused-variable"
            "-Wno-inconsistent-dllimport"
            "-Wno-unknown-pragmas"
            "-Wno-unused-parameter"
            "-Wno-sign-compare"
            "-Wno-deprecated-declarations"
            "-Wno-deprecated-dynamic-exception-spec"
            "-Wno-old-style-cast"
            "-Wno-sign-conversion"
            "-Wno-non-virtual-dtor"
            "-Wno-deprecated"
            "-Wno-shadow"
            "-Wno-shorten-64-to-32"
            "-Wno-reserved-id-macro"
            "-Wno-undef"
            "-Wno-c++98-compat-pedantic"
            "-Wno-documentation-unknown-command"
            "-Wno-zero-as-null-pointer-constant"
            "-Wno-language-extension-token"
            "-Wno-gnu-anonymous-struct"
            "-Wno-nested-anon-types"
            "-Wno-documentation"
            "-Wno-comma"
            "-Wno-nonportable-system-include-path"
            "-Wno-global-constructors"
            "-Wno-redundant-parens"
            "-Wno-exit-time-destructors"
            "-Wno-missing-noreturn"
            "-Wno-switch-enum"
            "-Wno-covered-switch-default"
            "-Wno-float-equal"
            "-Wno-double-promotion"
            "-Wno-microsoft-enum-value"
            "-Wno-missing-prototypes"
            "-Wno-implicit-fallthrough"
            "-Wno-format-nonliteral"
            "-Wno-cast-qual"
            "-Wno-disabled-macro-expansion"
            "-Wno-unused-private-field"
            "-Wno-unused-template"
            "-Wno-unused-macros"
            "-Wno-extra-semi-stmt"
            "-Wno-c++98-compat")
        list(APPEND MIZUBA_CXX_FLAGS_DEBUG ${_MIZUBA_CLANG_CL_DISABLED_WARNINGS})
        list(APPEND MIZUBA_CXX_FLAGS_RELEASE ${_MIZUBA_CLANG_CL_DISABLED_WARNINGS})
        unset(_MIZUBA_CLANG_CL_DISABLED_WARNINGS)
    else()
        # Problematic MSVC cl warnings.
        list(APPEND MIZUBA_CXX_FLAGS_DEBUG "/wd4459")
        list(APPEND MIZUBA_CXX_FLAGS_RELEASE "/wd4459")
    endif()
endif()
if(MINGW)
    # In MinGW some tests generate big object files.
    message(STATUS "Enabling the '-Wa,-mbig-obj' flag for MinGW.")
    list(APPEND MIZUBA_CXX_FLAGS_DEBUG "-Wa,-mbig-obj")
    list(APPEND MIZUBA_CXX_FLAGS_RELEASE "-Wa,-mbig-obj")
endif()

# Find the dependencies.

# heyoka.
find_package(heyoka 6.1.0 REQUIRED CONFIG)
message(STATUS "Found heyoka version ${heyoka_VERSION}")

# fmt.
find_package(fmt REQUIRED CONFIG)

# Mandatory dependency on TBB.
find_package(TBB REQUIRED CONFIG)

# Boost.
# NOTE: need 1.69 for safe numerics.
find_package(Boost 1.69 COMPONENTS serialization iostreams filesystem REQUIRED CONFIG)

# Python.
# Setup the list of Python components we need to locate.
set(_MIZUBA_PYTHON3_COMPONENTS Interpreter Development.Module)

# NOTE: strictly speaking, we should need only the Interpreter and Development.Module
# components to build mizuba. However, in certain setups (e.g., skbuild), it looks
# like if we specify only Development.Module CMake is unable to correctly locate
# the Python installation. Hence, as a workaround, if the following line fails
# and Python3 is *not* found, we try again, this time with the full Development
# component (rather than only Development.Module). This seems to work around the
# issue, for now at least.
find_package(Python3 QUIET COMPONENTS ${_MIZUBA_PYTHON3_COMPONENTS})

if(NOT Python3_FOUND)
    list(POP_BACK _MIZUBA_PYTHON3_COMPONENTS)
    list(APPEND _MIZUBA_PYTHON3_COMPONENTS Development)
endif()

find_package(Python3 QUIET REQUIRED COMPONENTS ${_MIZUBA_PYTHON3_COMPONENTS})

message(STATUS "Python3 interpreter: ${Python3_EXECUTABLE}")
message(STATUS "Python3 installation directory: ${Python3_SITEARCH}")
if(${Python3_VERSION_MINOR} LESS 7)
    message(FATAL_ERROR "The minimum supported version of Python is 3.7, but version 3.${Python3_VERSION_MINOR} was found instead")
endif()

set(MIZUBA_INSTALL_PATH "" CACHE STRING "mizuba module installation path")
mark_as_advanced(MIZUBA_INSTALL_PATH)
unset(_MIZUBA_PYTHON3_COMPONENTS)

# pybind11.
find_package(pybind11 REQUIRED CONFIG)
if(${pybind11_VERSION} VERSION_LESS "2.10")
    message(FATAL_ERROR "pybind11 >= 2.10 is required, but version ${pybind11_VERSION} was found instead.")
endif()

# Add the module directory.
add_subdirectory(mizuba)
