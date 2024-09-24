# Configure the version file.
configure_file("${CMAKE_CURRENT_SOURCE_DIR}/_version.py.in" "${CMAKE_CURRENT_BINARY_DIR}/_version.py" @ONLY)

# The list of mizuba's Python files.
set(MIZUBA_PYTHON_FILES
    __init__.py
    test/__init__.py
    test/test_sgp4_polyjectory.py
    test/_sgp4_test_data_202407.py
    test/_sgp4_test_data_202409.py
    test/test_conjunctions.py
    test/_planar_circ.py
    test/test_polyjectory.py
)

# Core module.
set(_MIZUBA_CORE_SOURCES
    core.cpp
    common_utils.cpp
    polyjectory.cpp
    sgp4_polyjectory.cpp
    conjunctions.cpp
    conjunctions_compute_aabbs.cpp
    detail/file_utils.cpp
)

Python3_add_library(core MODULE WITH_SOABI ${_MIZUBA_CORE_SOURCES})

unset(_MIZUBA_CORE_SOURCES)

target_link_libraries(core PRIVATE heyoka::heyoka fmt::fmt TBB::tbb Boost::boost Boost::serialization Boost::iostreams Boost::filesystem Boost::disable_autolinking)
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
target_compile_features(core PRIVATE cxx_std_20)
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
