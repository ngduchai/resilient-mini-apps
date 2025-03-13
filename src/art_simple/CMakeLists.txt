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
add_executable(art_simple main.cc tomo_recon.cc)
add_executable(art_simple_naive main-naive.cc tomo_recon.cc)
add_executable(art_simple_skip_frames main-frame-skip.cc tomo_recon.cc)
add_executable(art_simple_add_frames main-frame-add.cc tomo_recon.cc)
add_executable(art_simple_ckpt_quality_analysis main-ckpt-quality-analysis.cc tomo_recon.cc)
add_executable(art_simple_onoff main-onoff.cc tomo_recon.cc)
add_executable(art_simple_failure_impact main-failure-impact.cc tomo_recon.cc)
add_executable(art_simple_sync_recovery main-sync-recovery.cc tomo_recon.cc)
add_executable(art_simple_static_redis main-static-redis.cc tomo_recon.cc)
add_executable(art_simple_failures main-fixed-failures.cc tomo_recon.cc)
add_executable(art_simple_imbalance_impact main-imbalance-impact.cc tomo_recon.cc)
add_executable(art_simple_sync_recovery_imbalance main-sync-recovery-imbalance.cc tomo_recon.cc)
add_executable(art_simple_origin main-origin.cc tomo_recon.cc)

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
target_link_libraries(art_simple_origin HDF5::HDF5 veloc-client MPI::MPI_CXX)


