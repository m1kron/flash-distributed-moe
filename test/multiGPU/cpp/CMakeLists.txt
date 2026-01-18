set(app_name multiGPU_unittests_app)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${app_name} LANGUAGES CXX)

enable_language(HIP)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
find_package(HIP REQUIRED)
find_package(hsa-runtime64 REQUIRED)

include(FetchContent)
include(ExternalProject)

FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main)
FetchContent_MakeAvailable(googletest)


list(APPEND CMAKE_PREFIX_PATH "/opt/mpi/install/ompi")
find_package(MPI REQUIRED)

set(ROCSHMEM_INSTALL_DIR ${CMAKE_BINARY_DIR}/rocshmem-install)

# Create include directory at configure time to satisfy CMake validation
file(MAKE_DIRECTORY ${ROCSHMEM_INSTALL_DIR}/include)

ExternalProject_Add(
  rocshmem_external
  GIT_REPOSITORY https://github.com/ROCm/rocSHMEM.git
  GIT_TAG develop
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
add_library(rocshmem STATIC IMPORTED)
set_target_properties(rocshmem PROPERTIES
  IMPORTED_LOCATION ${ROCSHMEM_INSTALL_DIR}/lib/librocshmem.a
  INTERFACE_INCLUDE_DIRECTORIES ${ROCSHMEM_INSTALL_DIR}/include
)
add_dependencies(rocshmem rocshmem_external)

file(GLOB_RECURSE CPP_SOURCES ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HIP_SOURCES ${PROJECT_SOURCE_DIR}/*.hip)
set_source_files_properties(${HIP_SOURCES} ${CPP_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(${app_name} ${CPP_SOURCES} ${HIP_SOURCES})

target_include_directories(${app_name} PRIVATE ${FLASH_MOE_LIB_INCLUDE} ${ROCSHMEM_INSTALL_DIR}/include ${MPI_CXX_INCLUDE_PATH})
target_compile_definitions(${app_name} PRIVATE __HIP_PLATFORM_AMD__)

# rocSHMEM requires relocatable device code
target_compile_options(${app_name} PRIVATE -fgpu-rdc)
target_link_options(${app_name} PRIVATE -fgpu-rdc)

add_dependencies(${app_name} rocshmem_external)
target_link_libraries(${app_name} PRIVATE hip::host hip::device ${FLASH_MOE_LIB_NAME} gtest_main gtest rocshmem MPI::MPI_CXX hsa-runtime64::hsa-runtime64)