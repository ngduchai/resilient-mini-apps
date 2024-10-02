cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Set the project name
project(art_simple_main)

# Find the HDF5 library
find_package(HDF5 REQUIRED)

find_package(veloc REQUIRED)

find_package(MPI REQUIRED)
include_directories(${MPI_INCLUDE_PATH})

# Add an executable
add_executable(art_simple main.cc art_simple.cc)
add_executable(art_simple_skip_frames main-frame-skip.cc art_simple.cc)
add_executable(art_simple_add_frames main-frame-add.cc art_simple.cc)

# Link the HDF5 library
target_link_libraries(art_simple HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_skip_frames HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_add_frames HDF5::HDF5 veloc-client MPI::MPI_CXX)

