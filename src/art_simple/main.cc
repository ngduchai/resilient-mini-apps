#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include "art_simple.h"
#include "hdf5.h"

#include "veloc.hpp"

#include <mpi.h>

#include <unistd.h>
#include <limits.h>
#include <string.h>

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

// Save the reconstruction image as an HDF5 file
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

bool recover(veloc::client_t *ckpt, const char *name,  int sinogram_size, int &v, int *num_ckpt, int &numrows, float *recon, int *row_index) {
    ckpt->mem_protect(1, &numrows, sizeof(int), 1);
    v = ckpt->restart_test(name, v);
    if (v > 0) {
        ckpt->restart_begin(name, v);
        // Read # tasks and # row first
        ckpt->recover_mem(VELOC_RECOVER_SOME, {1});
        // Adjust the reconstruction area
        ckpt->mem_protect(0, num_ckpt, sizeof(int), 1);
        ckpt->mem_protect(2, recon, sizeof(float), numrows*sinogram_size);
        ckpt->mem_protect(3, row_index, sizeof(int), numrows);
        // Recover the data
        ckpt->recover_mem(VELOC_RECOVER_SOME, {0, 2, 3});
        ckpt->restart_end(true);
    }else{
        numrows = 0;
    }
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
    int sinogram_size = ngridx*ngridy;
    //int num_iter = 2;
    //int num_outer_iter = 5;
    //float center = 294.078;

    // swap axis in data dt dy
    float *data_swap = swapDimensions(data, dt, dy, dx, 0, 1);

    std::cout << "Completed reading the data, starting the reconstruction..." << std::endl;
    std::cout << "dt: " << dt << ", dy: " << dy << ", dx: " << dx << ", ngridx: " << ngridx << ", ngridy: " << ngridy << ", num_iter: " << num_iter << ", center: " << center << std::endl;

    const unsigned int recon_size = dy*ngridx*ngridy;
    float *recon = new float[recon_size];
    float *local_recon = new float[recon_size];
    float *local_data = new float[dx*dy*dt];
    int *row_indexes = new int[dy];


    /* Initiate MPI Communication */
    MPI_Init(&argc, &argv);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    int num_tasks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    const unsigned int mpi_root = 0;
    
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    std::cout << "Task ID " << id << " from " << hostname << std::endl;


    /* Initialize the working space */
    int num_rows = dy / num_tasks;
    int extra_rows = dy % num_tasks;
    int w_offset = num_rows*id + std::min(id, extra_rows);
    if (extra_rows != 0 && id < extra_rows) {
        num_rows++;
    }
    for (int i = 0; i < num_rows; ++i) {
        row_indexes[i] = w_offset + i;
    }
    memcpy(local_data, data_swap + w_offset*dt*dx, sizeof(float)*num_rows*dt*dx);

    // Initiate VeloC
    const char* ckpt_name = "art_simple";
    int num_ckpt = num_tasks;
    int progress = 0;

    // Check if there is are checkpoints from previous run
    veloc::client_t *ckpt = veloc::get_client((unsigned int)id, check_point_config);
    ckpt->mem_protect(0, &num_ckpt, sizeof(int), 1);
    if (id == mpi_root) {
        progress = ckpt->restart_test(ckpt_name, 0);
    }
    MPI_Bcast(&progress, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
    
    // Load the checkpoint if any 
    if (progress > 0) {
        recover(ckpt, ckpt_name, sinogram_size, progress, &num_ckpt, num_rows, local_recon, row_indexes);
    }

    // tracking active workers
    int tracker_size = std::max(num_tasks, num_ckpt);
    int * active_tracker = new int[tracker_size];
    int * prev_active_tracker = new int[tracker_size];
    for (int i = 0; i < num_ckpt; ++i) {
        active_tracker[i] = 1;
    }
    for (int i = num_ckpt; i < num_tasks; ++i) {
        active_tracker[i] = 0;
    }
    int task_is_active = 1;
    int active_tasks = num_tasks;

    // run the reconstruction
    while (progress < num_outer_iter) {

        // Sync the task status across all tasks
        std::swap(active_tracker, prev_active_tracker);
        MPI_Allgather(&task_is_active, 1, MPI_INT, active_tracker, 1, MPI_INT, MPI_COMM_WORLD);
        // Check for new and old tasks
        std::vector<int> added_tasks, removed_tasks;
        active_tasks = 0;
        int task_index = 0;
        for (int j = 0; j < tracker_size; ++j) {
            if (active_tracker[j] == 1 && prev_active_tracker[j] == 0) {
                added_tasks.push_back(j);
            }else if (active_tracker[j] == 0 && prev_active_tracker[j] == 1) {
                removed_tasks.push_back(j);
            }
            if (active_tracker[j] == 1) {
                if (j == id) {
                    task_index = active_tasks;
                }
                active_tasks++;
            }
        }

        if (!added_tasks.empty()) {
            // Some tasks are added, rebalance by moving some slices to new tasks
            int adj_num_rows = dy / active_tasks;
            int extra_rows = dy % active_tasks;
            int w_offset = adj_num_rows*task_index + std::min(id, extra_rows);
            if (extra_rows != 0 && task_index < extra_rows) {
                adj_num_rows++;
            }
            int transferred_rows = num_rows - adj_num_rows;
        }

        if (!removed_tasks.empty()) {
            int recovered_size = 0;
            float * recovered_recon = nullptr;
            int * recovered_row_indexes = nullptr;
            float * local_recovered_recon = nullptr;
            int * local_recovered_row_indexes = nullptr;
            if (task_is_active) {
                // Remainning active tasks recover the checkpoints of inactive ones
                int num_ckpt = (removed_tasks.size() / active_tasks);
                if (removed_tasks.size() % active_tasks > task_index) {
                    num_ckpt++;
                }

                local_recovered_recon = new float [recon_size];
                local_recovered_row_indexes = new int [dy];

                for (int j = 0; j < num_ckpt; ++j) {
                    unsigned int ckpt_id = removed_tasks[removed_tasks.size()*j + task_index];
                    
                    recovered_size += numrows;
                }
            }
            // Gather the checkpoints at the root then redistribute across tasks
            int total_recovered_size = 0;
            MPI_Allreduce(&recovered_size, &total_recovered_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            if (id == mpi_root) {
                recovered_recon = new float [total_recovered_size];
                recovered_row_indexes = new int [total_recovered_size];
            }
            // Sync with root
            MPI_Gather(local_recovered_recon, recovered_size, MPI_FLOAT, recovered_recon, recovered_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
            MPI_Gather(local_recovered_row_indexes, recovered_size, MPI_INT, recovered_recon, recovered_size, MPI_INT, mpi_root, MPI_COMM_WORLD);
            // Estimate the data size each task will receive
            if (task_is_active) {
                recovered_size = total_recovered_size / active_tasks;
                if (total_recovered_size % active_tasks > task_index) {
                    recovered_size++;
                }
            }else{
                recovered_size = 0;
            }
            // Reditribute data from root
            MPI_Scatter(recovered_recon, recovered_size, MPI_FLOAT, local_recovered_recon, recovered_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
            MPI_Scatter(recovered_row_indexes, recovered_size, MPI_FLOAT, local_recovered_row_indexes, recovered_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);

            // Make sure the rows are up to date
            art(w_data, w_dy, w_dt, w_dx, &center, theta, w_recon, w_ngridx, w_ngridy, num_iter);
        }

        std::cout<< "[task-" << id << "]: Outer iteration: " << i << std::endl;
        // art(data_swap, w_dy, w_dt, w_dx, &center, theta, w_recon, w_ngridx, w_ngridy, num_iter);
        art(local_data, num_rows, dt, dx, &center, theta, local_recon, ngridx, ngridy, num_iter);

        // Save progress with checkpoint
        progress++;
        num_ckpt = active_tasks;
        if (!ckpt->checkpoint(ckpt_name, progress)) {
            throw std::runtime_error("Checkpointing failured");
        }
        std::cout << "[task-" << id << "]: Checkpointed version " << i+1 << std::endl;

        
        

    }

    if (id == mpi_root) {
        std::cout << "Reconstructed data from workers" << std::endl;
    }
    MPI_Gather(w_recon, w_recon_size, MPI_FLOAT, recon, w_recon_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);

    const char * img_name = "recon.h5";
    if (id == mpi_root) {
        // write the reconstructed data to a file
        // Create the output file name
        std::ostringstream oss;
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
