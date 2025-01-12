# Configure the version file.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in" "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)

# The list of mizuba's Python files.
set(MIZUBA_PYTHON_FILES
    __init__.py
    _sgp4_polyjectory.py
    _dl_utils.py
    test/__init__.py
    test/_sgp4_test_data_20240705.py
    test/_sgp4_test_data_20240917.py
    test/_sgp4_test_data_20241026.py
    test/test_sgp4_polyjectory.py
    test/test_conjunctions.py
    test/test_boundary_conjunctions.py
    test/_planar_circ.py
    test/test_polyjectory.py
    test/test_heyoka_conjunctions.py
    test/test_data_sources.py
    test/test_make_sgp4_polyjectory.py
    test/single_gpe.parquet
    test/single_gpe_ds.parquet
    test/multi_gpes.parquet
    test/iss_gpes.parquet
    test/syncom_gpe.parquet
    data_sources/__init__.py
    data_sources/_common.py
    data_sources/_spacetrack.py
    data_sources/_celestrak.py
)

# Core module.
set(_MIZUBA_CORE_SOURCES
    core.cpp
    common_utils.cpp
    polyjectory.cpp
    sgp4_polyjectory.cpp
    conjunctions.cpp
    detect_conjunctions.cpp
    detect_conjunctions_aabbs.cpp
    detect_conjunctions_morton.cpp
    detect_conjunctions_bvh.cpp
    detect_conjunctions_broad_phase.cpp
    detect_conjunctions_narrow_phase.cpp
    make_sgp4_polyjectory.cpp
    logging.cpp
    detail/conjunctions_jit.cpp
    detail/file_utils.cpp
    detail/poly_utils.cpp
    detail/sgp4/SGP4_fix.cpp
)

Python3_add_library(core MODULE WITH_SOABI ${_MIZUBA_CORE_SOURCES})

unset(_MIZUBA_CORE_SOURCES)

target_link_libraries(core PRIVATE
    heyoka::heyoka
    fmt::fmt
    TBB::tbb
    spdlog::spdlog
    Boost::boost
    Boost::iostreams
    Boost::filesystem
    Boost::disable_autolinking
)
target_link_libraries(core PRIVATE "${pybind11_LIBRARIES}")
target_compile_definitions(core PRIVATE "BOOST_FILESYSTEM_VERSION=4")
target_include_directories(core SYSTEM PRIVATE "${pybind11_INCLUDE_DIR}" "${Python3_INCLUDE_DIRS}")
target_compile_definitions(core PRIVATE "${pybind11_DEFINITIONS}")
target_compile_options(core PRIVATE
    "$<$<CONFIG:Debug>:${MIZUBA_CXX_FLAGS_DEBUG}>"
    "$<$<CONFIG:Release>:${MIZUBA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:RelWithDebInfo>:${MIZUBA_CXX_FLAGS_RELEASE}>"
    "$<$<CONFIG:MinSizeRel>:${MIZUBA_CXX_FLAGS_RELEASE}>"
)
set_target_properties(core PROPERTIES CXX_VISIBILITY_PRESET hidden)
set_target_properties(core PROPERTIES VISIBILITY_INLINES_HIDDEN TRUE)
target_compile_features(core PRIVATE cxx_std_23)
set_property(TARGET core PROPERTY CXX_EXTENSIONS NO)

if (MIZUBA_ENABLE_IPO)
    include(CheckIPOSupported)
    check_ipo_supported(RESULT _MIZUBA_IPO_RESULT OUTPUT _MIZUBA_IPO_OUTPUT)
    if (_MIZUBA_IPO_RESULT)
        message(STATUS "IPO requested and supported, enabling.")
        set_property(TARGET core PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
    else()
        message(STATUS "IPO requested, but it is not supported by the compiler:\n${_MIZUBA_IPO_OUTPUT}")
    endif()
    unset(_MIZUBA_IPO_RESULT)
    unset(_MIZUBA_IPO_OUTPUT)
endif()

# Installation setup.
if(MIZUBA_INSTALL_PATH STREQUAL "")
    message(STATUS "mizuba will be installed in the default location: ${Python3_SITEARCH}")
    set(_MIZUBA_INSTALL_DIR "${Python3_SITEARCH}/mizuba")
else()
    message(STATUS "mizuba will be installed in the custom location: ${MIZUBA_INSTALL_PATH}")
    set(_MIZUBA_INSTALL_DIR "${MIZUBA_INSTALL_PATH}/mizuba")
endif()

# Install the core module.
install(TARGETS core
    RUNTIME DESTINATION ${_MIZUBA_INSTALL_DIR}
    LIBRARY DESTINATION ${_MIZUBA_INSTALL_DIR}
)

# Add the Python files.
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/_version.py" DESTINATION ${_MIZUBA_INSTALL_DIR})
foreach(_MIZUBA_CUR_FILE ${MIZUBA_PYTHON_FILES})
    get_filename_component(_MIZUBA_CUR_DIR ${_MIZUBA_CUR_FILE} DIRECTORY)
    install(FILES ${_MIZUBA_CUR_FILE} DESTINATION "${_MIZUBA_INSTALL_DIR}/${_MIZUBA_CUR_DIR}")
endforeach()

unset(_MIZUBA_INSTALL_DIR)
