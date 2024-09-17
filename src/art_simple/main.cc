#include <iostream>
#include <sstream>
#include <algorithm>
#include "art_simple.h"
#include "hdf5.h"

#include "veloc.hpp"

#include <mpi.h>

#include <unistd.h>
#include <limits.h>

// Function to swap dimensions of a flat 3D array
float* swapDimensions(float* original, int x, int y, int z, int dim1, int dim2) {
    float* transposed= new float[x*y*z];

    for (int i = 0; i < x; ++i) {
        for (int j = 0; j < y; ++j) {
            for (int k = 0; k < z; ++k) {
                int original_index = i * (y * z) + j * z + k;

                int transposed_index;
                if (dim1 == 1 && dim2 == 2) {  // Swap y and z
                    transposed_index = i * (z * y) + k * y + j;
                } else if (dim1 == 0 && dim2 == 2) {  // Swap x and z
                    transposed_index = k * (y * x) + j * x + i;
                } else if (dim1 == 0 && dim2 == 1) {  // Swap x and y
                    transposed_index = j * (x * z) + i * z + k;
                } else {
                    continue;  // No valid swap detected
                }

                transposed[transposed_index] = original[original_index];
            }
        }
    }

    return transposed;
}

int main(int argc, char* argv[])
{

    if(argc != 8) {
        std::cerr << "Usage: " << argv[0] << " <filename> <center> <num_outer_iter> <num_iter> <beginning_sino> <num_sino> [veloc config]" << std::endl;
        return 1;
    }

    std::cout << "argc: " << argc << std::endl;

    const char* filename = argv[1];
    float center = atof(argv[2]);
    int num_outer_iter = atoi(argv[3]);
    int num_iter = atoi(argv[4]);
    int beg_index = atoi(argv[5]);
    int nslices = atoi(argv[6]);
    const char* check_point_config = (argc == 8) ? argv[7] : "art_simple.cfg";

    std::cout << "Reading data..." << std::endl;

    // Open tomo_00058_all_subsampled1p_ HDF5 file
    //const char* filename = "../../data/tomo_00058_all_subsampled1p_s1079s1081.h5";
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return 1;
    }:q

    // Read the data from the HDF5 file
    const char* dataset_name = "exchange/data";
    hid_t dataset_id = H5Dopen(file_id, dataset_name, H5P_DEFAULT);
    if (dataset_id < 0) {
        std::cerr << "Error: Unable to open dataset " << dataset_name << std::endl;
        return 1;
    }

    //// read the data
    hid_t dataspace_id = H5Dget_space(dataset_id);
    hsize_t dims[3];
    H5Sget_simple_extent_dims(dataspace_id, dims, NULL);
    std::cout << "Data dimensions: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
    //float* data = new float[dims[0]*dims[1]*dims[2]];
    //H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    // read slices from the dataset
    std::cout << "Target dimensions: " << dims[0] << " x [" << beg_index << "-" << beg_index+nslices << "] x " << dims[2] << std::endl;
    hsize_t start[3] = {0, beg_index, 0};
    hsize_t count[3] = {dims[0], nslices, dims[2]};
    H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

    // Create a memory dataspace
    hsize_t mem_dims[3] = {dims[0], nslices, dims[2]};
    hid_t memspace_id = H5Screate_simple(3, mem_dims, NULL);

    // Allocate memory for the hyperslab
    float* data = new float[dims[0] * nslices * dims[2]];

    // Read the data from the hyperslab
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id, H5P_DEFAULT, data);

    //close the dataset
    H5Dclose(dataset_id);
    H5Sclose(memspace_id);

    // read the theta
    const char* theta_name = "exchange/theta";
    hid_t theta_id = H5Dopen(file_id, theta_name, H5P_DEFAULT);
    if (theta_id < 0) {
        std::cerr << "Error: Unable to open dataset " << theta_name << std::endl;
        return 1;
    }
    // read the data
    hid_t theta_dataspace_id = H5Dget_space(theta_id);
    hsize_t theta_dims[1];
    H5Sget_simple_extent_dims(theta_dataspace_id, theta_dims, NULL);
    std::cout << "Theta dimensions: " << theta_dims[0] << std::endl;
    float* theta = new float[theta_dims[0]];
    H5Dread(theta_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, theta);
    // close the dataset
    H5Dclose(theta_id);

    // Close the HDF5 file
    H5Fclose(file_id);

    // reconstruct using art
    int dt = dims[0];
    int dy = nslices; //dims[1];
    int dx = dims[2];
    int ngridx = dx;
    int ngridy = dx;
    //int num_iter = 2;
    //int num_outer_iter = 5;
    //float center = 294.078;

    // swap axis in data dt dy
    float *data_swap = swapDimensions(data, dt, dy, dx, 0, 1);

    std::cout << "Completed reading the data, starting the reconstruction..." << std::endl;
    std::cout << "dt: " << dt << ", dy: " << dy << ", dx: " << dx << ", ngridx: " << ngridx << ", ngridy: " << ngridy << ", num_iter: " << num_iter << ", center: " << center << std::endl;

    const unsigned int recon_size = dy*ngridx*ngridy;
    float *recon = new float[recon_size];

    /* Initiate MPI Communication */
    MPI_Init(&argc, &argv);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    int num_workers;
    MPI_Comm_size(MPI_COMM_WORLD, &num_workers);
    const unsigned int mpi_root = 0;
    
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    std::cout << "Task ID " << id << " from " << hostname << std::endl; 

    /* Calculating the working area based on worker id */
    int rows_per_worker = dy / num_workers;
    int extra_rows = dy % num_workers;
    int w_offset = rows_per_worker*id + std::min(id, extra_rows);
    if (extra_rows != 0 && id < extra_rows) {
        rows_per_worker++;
    }
    hsize_t w_dt = dt;
    hsize_t w_dy = rows_per_worker;
    hsize_t w_dx = dx;
    hsize_t w_ngridx = ngridx;
    hsize_t w_ngridy = ngridy;
    
    const unsigned int w_recon_size = rows_per_worker*ngridx*ngridy;
    // float * w_recon = recon + w_offset*ngridx*ngridy;
    float * w_recon = new float [w_recon_size];
    float * w_data = data_swap + w_offset*dt*dx;

    std::cout << "[task-" << id << "]: offset: " << w_offset << ", w_dt: " << w_dt << ", w_dy: " << w_dy << ", w_dx: " << w_dx << ", w_ngridx: " << w_ngridx << ", w_ngridy: " << w_ngridy << ", num_iter: " << num_iter << ", center: " << center << std::endl;

    // Initiate VeloC
    veloc::client_t *ckpt = veloc::get_client((unsigned int)id, check_point_config);

    ckpt->mem_protect(0, w_recon, sizeof(float), w_recon_size);
    const char* ckpt_name = "art_simple";

    int v = ckpt->restart_test(ckpt_name, 0);
    if (v > 0) {
        std::cout << "[task-" << id << "]: Found a checkpoint version " << v << " at iteration #" << v-1 << ", initiating restart" << std::endl;
        if (!ckpt->restart(ckpt_name, v)) {
            throw std::runtime_error("restart failed");
        }
    }else {
        v = 0;
    }
    std::cout << "[task-" << id << "]: Start the reconstruction from iteration #" << v << std::endl;

    // run the reconstruction
    for (int i = v; i < num_outer_iter; i++)
    {
        // std::cout << "Outer iteration: " << i << std::endl;
        // art(data_swap, w_dy, w_dt, w_dx, &center, theta, w_recon, w_ngridx, w_ngridy, num_iter);
        art(w_data, w_dy, w_dt, w_dx, &center, theta, w_recon, w_ngridx, w_ngridy, num_iter);

        // Checkpointing
        std::cout << "[task-" << id << "]: Checkpointing for version " << i+1 << std::endl;
        if (!ckpt->checkpoint(ckpt_name, i+1)) {
            throw std::runtime_error("Checkpointing failured");
        }
        std::cout << "[task-" << id << "]: Checkpointed version " << i+1 << std::endl;

        // Also push the result to disk for further quality analysis
        MPI_Gather(w_recon, w_recon_size, MPI_FLOAT, recon, w_recon_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
        if (id == mpi_root) {
            
            std::ostringstream oss;
            oss << "recon_tmp_" << std::setw(4) << std::setfill('0') << i << ".h5";
            std::cout << "Saving a temporary reconstructed image as" << oss.str() << std::endl;
            std::string output_filename = oss.str();
            const char* output_filename_cstr = output_filename.c_str();

            hid_t output_file_id = H5Fcreate(output_filename_cstr, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
            if (output_file_id < 0) {
                std::cerr << "Error: Unable to create file " << output_filename << std::endl;
                return 1;
            }
            hsize_t output_dims[3] = {dy, ngridy, ngridx};
            hid_t output_dataspace_id = H5Screate_simple(3, output_dims, NULL);
            hid_t output_dataset_id = H5Dcreate(output_file_id, "/data", H5T_NATIVE_FLOAT, output_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
            H5Dwrite(output_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, recon);
            H5Dclose(output_dataset_id);
            H5Sclose(output_dataspace_id);
            H5Fclose(output_file_id);
            std::cout << "Saved a temporary reconstructed image as" << oss.str() << std::endl;
        }

    }

    }

    if (id == mpi_root) {
        std::cout << "reconstructed data from workers" << std::endl;
    }
    MPI_Gather(w_recon, w_recon_size, MPI_FLOAT, recon, w_recon_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);

    const char * img_name = "recon.h5";
    if (id == mpi_root) {
        std::cout << "Save the reconstruction image as " << img_name << std::endl;

        // write the reconstructed data to a file
        // Create the output file name
        std::ostringstream oss;
        // if (check_point_path != nullptr) {
        //     oss << "cont_recon_" << i << ".h5";
        // } else {
        //     oss << "recon_" << i << ".h5";
        // }
        oss << img_name;

        std::string output_filename = oss.str();
        const char* output_filename_cstr = output_filename.c_str();

        hid_t output_file_id = H5Fcreate(output_filename_cstr, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
        if (output_file_id < 0) {
            std::cerr << "Error: Unable to create file " << output_filename << std::endl;
            return 1;
        }
        hsize_t output_dims[3] = {dy, ngridy, ngridx};
        hid_t output_dataspace_id = H5Screate_simple(3, output_dims, NULL);
        hid_t output_dataset_id = H5Dcreate(output_file_id, "/data", H5T_NATIVE_FLOAT, output_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
        H5Dwrite(output_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, recon);
        H5Dclose(output_dataset_id);
        H5Sclose(output_dataspace_id);
        H5Fclose(output_file_id);

    }


    // free the memory
    delete[] data;
    delete[] data_swap;
    delete[] theta;
    delete[] recon;
    delete[] w_recon;

    MPI_Finalize();

    return 0;
}
