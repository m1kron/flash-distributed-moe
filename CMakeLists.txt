cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
set(FLASH_MOE_LIB_NAME flashMoe)

project(${FLASH_MOE_LIB_NAME} LANGUAGES CXX)

enable_language(HIP)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")

set(ROCM_ROOT
    "/opt/rocm"
    CACHE PATH
    "Root directory of the ROCm installation"
)

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
find_package(HIP REQUIRED)

add_link_options(-Wl,--version-script=${PROJECT_SOURCE_DIR}/src/lib.version)

file(GLOB_RECURSE HIP_SRC ${PROJECT_SOURCE_DIR}/src/*.hip)
file(GLOB_RECURSE CPP_SRC ${PROJECT_SOURCE_DIR}/src/*.cpp)

add_library(${FLASH_MOE_LIB_NAME} SHARED ${HIP_SRC} ${CPP_SRC})
target_include_directories(${FLASH_MOE_LIB_NAME} PRIVATE ${PROJECT_SOURCE_DIR})
target_link_libraries(${FLASH_MOE_LIB_NAME} PRIVATE hip::host)
set_source_files_properties(${HIP_SRC} PROPERTIES LANGUAGE HIP)
set_target_properties(${FLASH_MOE_LIB_NAME} PROPERTIES POSITION_INDEPENDENT_CODE ON)

set(FLASH_MOE_LIB_INCLUDE ${PROJECT_SOURCE_DIR}) 

# Build unittests:
add_subdirectory(test/singleGPU/cpp/)

add_subdirectory(test/multiGPU/cpp/)
