# BuildRocshmem.cmake
# Reusable CMake module for building rocSHMEM as an external project
#
# Usage:
#   include(BuildRocshmem)
#   build_rocshmem([GIT_TAG develop])
#
# After calling build_rocshmem(), the following targets and variables are available:
#   - rocshmem (imported static library target)
#   - rocshmem_INCLUDE_DIRS

include(ExternalProject)

function(build_rocshmem)
  set(options "")
  set(oneValueArgs GIT_TAG)
  set(multiValueArgs "")
  cmake_parse_arguments(ROCSHMEM_BUILD "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  # Set defaults
  set(ROCSHMEM_INSTALL_DIR ${CMAKE_BINARY_DIR}/_deps/rocshmem-install)

  if(NOT ROCSHMEM_BUILD_GIT_TAG)
    set(ROCSHMEM_BUILD_GIT_TAG develop)
  endif()

  # Find required dependencies
  find_package(MPI REQUIRED)
  find_package(hsa-runtime64 REQUIRED)

  # Create include directory at configure time to satisfy CMake validation
  file(MAKE_DIRECTORY ${ROCSHMEM_INSTALL_DIR}/include)

  ExternalProject_Add(
    rocshmem_external
    GIT_REPOSITORY https://github.com/ROCm/rocSHMEM.git
    GIT_TAG ${ROCSHMEM_BUILD_GIT_TAG}
    CMAKE_ARGS
      -DCMAKE_INSTALL_PREFIX=${ROCSHMEM_INSTALL_DIR}
      -DBUILD_CODE_COVERAGE=OFF
      -DCMAKE_BUILD_TYPE=Release
      -DCMAKE_VERBOSE_MAKEFILE=OFF
      -DCMAKE_POSITION_INDEPENDENT_CODE=ON
      -DBUILD_FUNCTIONAL_TESTS=OFF
      -DBUILD_UNIT_TESTS=OFF
      -DBUILD_TOOLS=OFF
      -DBUILD_EXAMPLES=OFF
      -DDEBUG=OFF
      -DPROFILE=OFF
      -DUSE_GDA=OFF
      -DUSE_RO=OFF
      -DUSE_IPC=ON
      -DUSE_THREADS=OFF
      -DUSE_WF_COAL=OFF
      -DUSE_HDP_FLUSH=OFF
      -DUSE_HDP_FLUSH_HOST_SIDE=OFF
      -DUSE_SINGLE_NODE=ON
    BUILD_BYPRODUCTS ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a
  )

  # Create imported target for rocshmem
  add_library(rocshmem STATIC IMPORTED GLOBAL)
  set_target_properties(rocshmem PROPERTIES
    IMPORTED_LOCATION ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a
    INTERFACE_INCLUDE_DIRECTORIES ${ROCSHMEM_INSTALL_DIR}/include
    INTERFACE_LINK_LIBRARIES "MPI::MPI_CXX;hip::device;hip::host;dl;hsa-runtime64::hsa-runtime64;-fgpu-rdc"
    INTERFACE_COMPILE_OPTIONS "-fgpu-rdc"
  )
  add_dependencies(rocshmem rocshmem_external)

  # Export variables to parent scope
  set(rocshmem_INCLUDE_DIRS ${ROCSHMEM_INSTALL_DIR}/include PARENT_SCOPE)
  set(rocshmem_FOUND TRUE PARENT_SCOPE)

endfunction()
