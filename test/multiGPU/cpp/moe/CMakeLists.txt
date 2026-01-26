set(app_name multiGPU_unittests_app)

cmake_minimum_required(VERSION 3.21 FATAL_ERROR)
project(${app_name} LANGUAGES CXX)

enable_language(HIP)
set(CMAKE_HIP_STANDARD 17)
set(CMAKE_HIP_EXTENSIONS OFF)
set(CMAKE_HIP_STANDARD_REQUIRED ON)

list(APPEND CMAKE_PREFIX_PATH "${ROCM_ROOT}")
find_package(HIP REQUIRED)

include(FetchContent)
FetchContent_Declare(
  googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG main)
FetchContent_MakeAvailable(googletest)

file(GLOB_RECURSE CPP_SOURCES ${PROJECT_SOURCE_DIR}/*.cpp)
file(GLOB_RECURSE HIP_SOURCES ${PROJECT_SOURCE_DIR}/*.hip)
set_source_files_properties(${HIP_SOURCES} PROPERTIES LANGUAGE HIP)
add_executable(${app_name} ${CPP_SOURCES} ${HIP_SOURCES})
target_include_directories(${app_name} PRIVATE ${FLASH_MOE_LIB_INCLUDE})
target_compile_definitions(${app_name} PRIVATE __HIP_PLATFORM_AMD__)
target_link_libraries(${app_name} PRIVATE hip::host ${FLASH_MOE_LIB_NAME} gtest_main gtest)