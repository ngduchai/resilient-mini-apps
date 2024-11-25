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
add_executable(art_simple_naive main-naive.cc art_simple.cc)
add_executable(art_simple_skip_frames main-frame-skip.cc art_simple.cc)
add_executable(art_simple_add_frames main-frame-add.cc art_simple.cc)
add_executable(art_simple_ckpt_quality_analysis main-ckpt-quality-analysis.cc art_simple.cc)
add_executable(art_simple_onoff main-onoff.cc art_simple.cc)
add_executable(art_simple_failure_impact main-failure-impact.cc art_simple.cc)
add_executable(art_simple_sync_recovery main-sync-recovery.cc art_simple.cc)
add_executable(art_simple_static_redis main-static-redis.cc art_simple.cc)
add_executable(art_simple_failures main-fixed-failures.cc art_simple.cc)
add_executable(art_simple_imbalance_impact main-imbalance-impact.cc art_simple.cc)
add_executable(art_simple_sync_recovery_imbalance main-sync-recovery-imbalance.cc art_simple.cc)

# Link the HDF5 library
target_link_libraries(art_simple HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_naive HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_skip_frames HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_add_frames HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_ckpt_quality_analysis HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_onoff HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_failure_impact HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_sync_recovery HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_static_redis HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_failures HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_imbalance_impact HDF5::HDF5 veloc-client MPI::MPI_CXX)
target_link_libraries(art_simple_sync_recovery_imbalance HDF5::HDF5 veloc-client MPI::MPI_CXX)


