#include <iostream>
#include <sstream>
#include <algorithm>
#include "art_simple.h"
#include "hdf5.h"
#include <chrono>
#include <string.h>

#include <veloc.hpp>

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

int saveAsHDF5(const char* fname, float* recon, hsize_t* output_dims) {
    hid_t output_file_id = H5Fcreate(fname, H5F_ACC_TRUNC, H5P_DEFAULT, H5P_DEFAULT);
    if (output_file_id < 0) {
        return 1;
    }
    hid_t output_dataspace_id = H5Screate_simple(3, output_dims, NULL);
    hid_t output_dataset_id = H5Dcreate(output_file_id, "/data", H5T_NATIVE_FLOAT, output_dataspace_id, H5P_DEFAULT, H5P_DEFAULT, H5P_DEFAULT);
    H5Dwrite(output_dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, recon);
    H5Dclose(output_dataset_id);
    H5Sclose(output_dataspace_id);
    H5Fclose(output_file_id);
    return 0;
}

int main(int argc, char* argv[])
{
    std::cout << "argc: " << argc << std::endl;

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

    // Open tomo_00058_all_subsampled1p_ HDF5 file
    //const char* filename = "../../data/tomo_00058_all_subsampled1p_s1079s1081.h5";
    hid_t file_id = H5Fopen(filename, H5F_ACC_RDONLY, H5P_DEFAULT);
    if (file_id < 0) {
        std::cerr << "Error: Unable to open file " << filename << std::endl;
        return 1;
    }

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
    float* data = new float[dims[0]*dims[1]*dims[2]];
    H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

    //close the dataset
    H5Dclose(dataset_id);

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
    int dy = dims[1];
    int dx = dims[2];
    int ngridx = dx;
    int ngridy = dx;
    //int num_iter = 2;
    //int num_outer_iter = 5;
    //float center = 294.078;
    // float *recon = new float[dy*ngridx*ngridy];

    // swap axis in data dt dy
    float *data_swap = swapDimensions(data, dt, dy, dx, 0, 1);

    // duplicate the data to adjust the dy slices with nslices if needed
    if (nslices > dy) {
        std::cout << "nslices = " << nslices << " > " << dy << " = dy -- start data duplication" << std::endl; 
        float * data_tmp = data_swap;

        data_swap = new float [dx*nslices*dt];
        int additional_slices = nslices - dy;
        int i = 1;
        while (additional_slices > dy) {
            memcpy(data_swap + i*dx*dt*dy, data_tmp, dx*dt*dy);
            additional_slices -= dy;
            i++;
        }
        if (additional_slices > 0) {
            memcpy(data_swap + i*dx*dt*dy, data_tmp, dx*dt*additional_slices);
        }
        delete [] data_tmp;
    }
    dy = nslices;

    const unsigned int recon_size = dy*ngridx*ngridy;
    float *recon = new float[recon_size];

    std::cout << "dt: " << dt << ", dy: " << dy << ", dx: " << dx << 
                 ", ngridx: " << ngridx << ", ngridy: " << ngridy << 
                 ", num_iter: " << num_iter << ", center: " << center << 
                 ", beg_index: " << beg_index <<
                 ", nslices: " << nslices << std::endl;

    // Initiate VeloC
    unsigned int id = 0;
    veloc::client_t *ckpt = veloc::get_client(id, check_point_config);

    // ckpt->mem_protect(0, w_recon, sizeof(float), w_recon_size);
    ckpt->mem_protect(0, recon, sizeof(float), recon_size);
    const char* ckpt_name = "art_simple";

    auto recon_start = std::chrono::high_resolution_clock::now();

    // run the reconstruction
    for (int i = 0; i < num_outer_iter; i++)
    {
        std::cout << "Outer iteration: " << i << std::endl;
        art(data_swap, dy, dt, dx, &center, theta, recon, ngridx, ngridy, num_iter);

        if (!ckpt->checkpoint(ckpt_name, i+1)) {
            throw std::runtime_error("Checkpointing failured");
        }
    }

    auto recon_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> recon_elapsed = recon_end - recon_start;

    std::cout << "ELAPSED TIME: " << recon_elapsed.count() << " seconds" << std::endl;

    const char * img_name = "recon.h5";
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

    hsize_t output_dims[3] = {dy, ngridy, ngridx};
    if (saveAsHDF5(output_filename_cstr, recon, output_dims) != 0) {
        std::cerr << "Error: Unable to create file " << output_filename << std::endl;
        return 1;
    }
    else{
        std::cout << "Save the reconstruction image as " << img_name << std::endl;
    }

    // free the memory
    delete[] data;
    delete[] data_swap;
    delete[] theta;
    delete[] recon;

    return 0;
}