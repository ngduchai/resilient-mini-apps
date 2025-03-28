#include <iostream>
#include <sstream>
#include <algorithm>
#include <iomanip>
// #include "art_simple.h"
#include "tomo_recon.h"
#include "hdf5.h"

#include "veloc.hpp"

#include <mpi.h>

#include <unistd.h>
#include <limits.h>
#include <string.h>
#include <random>
#include <chrono>
#include <fstream>
#include <cassert>
#include <thread>

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

void recover(veloc::client_t *ckpt, int id, const char *name,  int sinogram_size, int &progress, int *num_ckpt, int &numrows, float *recon, int *row_index, int *row_progress, bool exact_ver=false) {
    // ckpt->mem_protect(1, &numrows, 1, sizeof(int));
    // int v = ckpt->restart_test(name, progress, id);
    // std::cout << "Checkpoint: " << id << " -- v = " << v << std::endl;
    // if (v > 0) {
    //     ckpt->restart_begin(name, v, id);
    //     // Read # tasks and # row first
    //     ckpt->recover_mem(VELOC_RECOVER_SOME, {1});
    //     // Adjust the reconstruction area
    //     ckpt->mem_protect(0, num_ckpt, 1, sizeof(int));
    //     ckpt->mem_protect(2, recon, numrows*sinogram_size, sizeof(float));
    //     ckpt->mem_protect(3, row_index, numrows, sizeof(int));
    //     // Recover the data
    //     ckpt->recover_mem(VELOC_RECOVER_SOME, {0, 2, 3});
    //     ckpt->restart_end(true);
    //     progress = v;
    // }else{
    //     numrows = 0;
    // }
    ckpt->mem_protect(0, num_ckpt, 1, sizeof(int));
    ckpt->mem_protect(1, &numrows, 1, sizeof(int));
    ckpt->mem_protect(2, recon, numrows*sinogram_size, sizeof(float));
    ckpt->mem_protect(3, row_index, numrows, sizeof(int));
    ckpt->mem_protect(4, row_progress, numrows, sizeof(int));
    
    // if (ckpt->restart(name, progress, id) == false) {
    //     numrows = 0;
    // }
    bool loaded = false;
    do {
        loaded = ckpt->restart(name, progress, id);
        if (loaded == false) {
            progress--;
        }
    } while (!exact_ver && loaded == false && progress >= 0);
    if (loaded == false) {
        numrows = 0;
    }
}

int main(int argc, char* argv[])
{

    if(argc != 11) {
        std::cerr << "Usage: " << argv[0] << " <filename> <center> <num_outer_iter> <num_iter> <beginning_sino> <num_sino> <failure_prob> <allow_restart> <recon_method> [veloc config]" << std::endl;
        return 1;
    }

    std::cout << "argc: " << argc << std::endl;

    const char* filename = argv[1];
    float center = atof(argv[2]);
    int num_outer_iter = atoi(argv[3]);
    int num_iter = atoi(argv[4]);
    int beg_index = atoi(argv[5]);
    int nslices = atoi(argv[6]);
    float failure_prob = atof(argv[7]);
    bool allow_restart = atoi(argv[8]) == 1 ? true : false;
    std::string recon_method(argv[9]);
    const char* check_point_config = (argc == 11) ? argv[10] : "art_simple.cfg";

    std::cout << "Reconstruction Method: " << recon_method << std::endl;

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

    // std::cout << "Data dimensions: " << dims[0] << " x " << dims[1] << " x " << dims[2] << std::endl;
    
    float* data = nullptr;
    if (dims[1] <= nslices) {
        // We will process the whole dataset so just read the whole file
        data  = new float[dims[0]*dims[1]*dims[2]]; 
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, H5S_ALL, H5S_ALL, H5P_DEFAULT, data);

        //close the dataset
        H5Dclose(dataset_id);
    }else{
        // We just process a part of the dataset, so we will read the region defined by [begining_sino, beginning_sino+nslices)
        hsize_t start[3] = {0, beg_index, 0};
        hsize_t count[3] = {dims[0], nslices, dims[2]};
        H5Sselect_hyperslab(dataspace_id, H5S_SELECT_SET, start, NULL, count, NULL);

        // Create a memory dataspace
        hsize_t mem_dims[3] = {dims[0], nslices, dims[2]};
        hid_t memspace_id = H5Screate_simple(3, mem_dims, NULL);

        // Allocate memory for the hyperslab
        data = new float[dims[0] * nslices * dims[2]];

        // Read the data from the hyperslab
        H5Dread(dataset_id, H5T_NATIVE_FLOAT, memspace_id, dataspace_id, H5P_DEFAULT, data);

        //close the dataset
        H5Dclose(dataset_id);
        H5Sclose(memspace_id);
    }

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
    int dy = std::min((hsize_t)nslices, dims[1]);
    int dx = dims[2];
    int ngridx = dx;
    int ngridy = dx;
    int sinogram_size = ngridx*ngridy;
    //int num_iter = 2;
    //int num_outer_iter = 5;
    //float center = 294.078;

    // swap axis in data dt dy
    float *data_swap = swapDimensions(data, dt, dy, dx, 0, 1);


    // duplicate the data to adjust the dy slices with nslices if needed
    if (nslices > dy) {
        float * data_tmp = data_swap;

        data_swap = new float [dx*nslices*dt];
        int additional_slices = nslices;
        int i = 0;
        while (additional_slices > dy) {
            memcpy(data_swap + i*dx*dt*dy, data_tmp, sizeof(float)*dx*dt*dy);
            additional_slices -= dy;
            i++;
        }
        if (additional_slices > 0) {
            memcpy(data_swap + i*dx*dt*dy, data_tmp, sizeof(float)*dx*dt*additional_slices);
        }
        delete [] data_tmp;
    }
    dy = nslices;

    // scale down the data by 2x
    std::cout << "Adjust dimentions" << std::endl;
    int original_dx = dx;
    int original_dt = dt;
    dx /= 2;
    dy /= 2;
    ngridx = dx;
    ngridy = dx;
    sinogram_size = ngridx*ngridy;
    float *data_tmp = new float[dx*dt*dy];
    for (int i = 0; i < dy; ++i) {
        for (int j = 0; j < dx; ++j) {
            for (int k = 0; k < dt; ++k) {
                data_tmp[i*dx*dt + j*dt + k] = (data_swap[i*original_dx*original_dt + j*2*original_dt + 2*k] + data_swap[i*original_dx*original_dt + j*2*original_dt + 2*k+1] + data_swap[i*original_dx*original_dt + (j*2+1)*original_dt + 2*k] + data_swap[i*original_dx*original_dt + (j*2+1)*original_dt + 2*k+1]) / 4;
            }
        }
    }
    delete [] data_swap;
    data_swap = data_tmp;



    std::cout << "Completed reading the data, starting the reconstruction..." << std::endl;
    std::cout << "dt: " << dt << ", dy: " << dy << ", dx: " << dx << ", ngridx: " << ngridx << ", ngridy: " << ngridy << ", num_iter: " << num_iter << ", center: " << center << std::endl;

    const unsigned int recon_size = dy*ngridx*ngridy;
    float *recon = new float[recon_size];
    float *local_recon = new float[recon_size];
    float *local_data = new float[dx*dy*dt];
    int *row_indexes = new int[dy];
    int *local_progress = new int[recon_size];

    // Ensure recon is initialized for MLEM
    if (recon_method == "mlem") {
        memset(local_recon, 1, recon_size * sizeof(float));
    }


    /* Initiate MPI Communication */
    MPI_Init(&argc, &argv);
    int id;
    MPI_Comm_rank(MPI_COMM_WORLD, &id);
    int num_tasks;
    MPI_Comm_size(MPI_COMM_WORLD, &num_tasks);
    unsigned int mpi_root = 0;
    
    char hostname[HOST_NAME_MAX];
    gethostname(hostname, HOST_NAME_MAX);
    pid_t process_id = getpid();
    std::cout << "Task ID " << id << " from " << hostname << ":" << process_id << std::endl;


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
    std::cout << id << ": " << num_rows << " -- offset: " << w_offset << std::endl;

    // Initiate VeloC
    const char* ckpt_name = "art_simple";
    int num_ckpt = num_tasks;
    int progress = 0;

    // Check if there is are checkpoints from previous run
    // veloc::client_t *ckpt = veloc::get_client((unsigned int)id, check_point_config);
    veloc::client_t *ckpt = veloc::get_client((unsigned int)id, check_point_config);
    ckpt->mem_protect(0, &num_ckpt, 1, sizeof(int));

    // ckpt->checkpoint(ckpt_name, 1);

    // ckpt = veloc::get_client((unsigned int)10, check_point_config);
    // ckpt->checkpoint(ckpt_name, 2);

    // if (progress == 0) {
    //     return 0;
    // }

    // std::vector<double> task_stop_thresholds = {210.968, 96.734, 194.45, 268.688, 21.9463, 188.592, 184.785, 9.77297, 11.4375, 59.2791};
    // std::vector<double> task_stop_thresholds = {210.968, 196.734, 194.45, 268.688, 21.9463, 188.592, 184.785, 9.77297, 11.4375, 59.2791};
    // std::vector<double> task_stop_thresholds = {1210.968, 1196.734, 1194.45, 1268.688, 1121.9463, 1188.592, 1184.785, 1119.77297, 1111.4375, 1159.2791};
    // std::vector<double> task_stop_thresholds = {10.5663, 151.73, 276.904, 25.1491};
    // double task_stop_threshold = task_stop_thresholds[id];
    // Setup random number generation
    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::exponential_distribution exp_dist(failure_prob); // Distribution for 0 and 1
    double task_stop_threshold = exp_dist(gen);
    std::cout << "[Task-" << id << "] will stop in the next " << task_stop_threshold << " second(s)." << std::endl;

    double reconstruction_time_per_iter = 0.0;
    bool found_crashed = false;

    // std::vector<std::vector<int>> task_states = {
    //     {1, 0, 0, 1, 0, 0, 0, 1, 1, 1},
    //     {0, 0, 0, 1, 0, 0, 0, 0, 1, 1},
    //     {1, 1, 0, 0, 1, 1, 0, 1, 1, 1},
    //     {0, 1, 0, 1, 0, 0, 0, 1, 1, 0},
    //     {1, 0, 1, 0, 1, 1, 1, 0, 0, 1},
    // };
    // std::vector<std::vector<int>> task_states = {
    //     {1, 0, 1, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 1, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 1, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 1, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 1, 0, 0, 0, 1, 0, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 1, 0, 0},
    //     {1, 0, 0, 0, 0, 0, 0, 0, 0, 0},
    //     {1, 0, 0, 1, 0, 0, 0, 0, 0, 0}
    // };
    std::vector<int> task_state_history;
    int stat_num_failures = 0;
    
    if (id == mpi_root) {
        progress = ckpt->restart_test(ckpt_name, 0, id);
    }
    MPI_Bcast(&progress, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
    
    // Load the checkpoint if any 
    if (progress > 0) {
        std::cout << "[Task-" << id << "]: Recover checkpoint " << id << " from progress " << progress << std::endl;
        num_rows = dy;
        recover(ckpt, id, ckpt_name, sinogram_size, progress, &num_ckpt, num_rows, local_recon, local_progress, row_indexes);
        for (int i = 0; i < num_rows; ++i) {
            memcpy(local_data + i*dt*dx, data_swap+row_indexes[i]*dt*dx, sizeof(float)*dt*dx);
        }
        std::cout << "[Task-" << id << "]: Recovery completed, checkpoint: " << id << ", progress: " << progress << " num_row: " << num_rows << " num_ckpt: " << num_ckpt << std::endl;
        MPI_Bcast(&num_ckpt, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
    }else{
        progress = 0;
        ckpt->mem_protect(1, &num_rows, 1, sizeof(int));
        ckpt->mem_protect(2, local_recon, num_rows*sinogram_size, sizeof(float));
        ckpt->mem_protect(3, row_indexes, num_rows, sizeof(int));
        ckpt->mem_protect(4, local_progress, num_rows, sizeof(int));
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
    for (int i = 0; i < tracker_size; ++i) {
        prev_active_tracker[i] = 0;
    }
    int task_is_active = 1;
    int active_tasks = num_tasks;
    // Index to the sinograms that still needs to be processed.
    int local_active_index = 0;

    double ckpt_time = 0;
    double recovery_time = 0;
    double comm_time = 0;
    double exec_time = 0;
    auto recon_start = std::chrono::high_resolution_clock::now();
    auto exec_start = std::chrono::high_resolution_clock::now();
    auto exec_end = std::chrono::high_resolution_clock::now();

    // run the reconstruction
    // while (progress < num_outer_iter+1) {
    while (true) {
        // We need one extra iteration to ensure all rows are sync with the expected number of iterations
        
        // MPI_Barrier(MPI_COMM_WORLD);

        // Sync the task status across all tasks
        std::swap(active_tracker, prev_active_tracker);
        MPI_Allgather(&task_is_active, 1, MPI_INT, active_tracker, 1, MPI_INT, MPI_COMM_WORLD);
        
        // The allgather also serve as a synchronization barrier that let us know
        // also tasks complete their computation. This is the good moment to
        // know that the reconstruction of the the previous iteration is completed
        // and we start to establish communication to redistribute data if needed.
        exec_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = exec_end - exec_start;
        exec_time += elapsed_time.count();
        auto comm_start = std::chrono::high_resolution_clock::now();
        
        // Check for new and old tasks
        std::vector<int> added_tasks, removed_tasks;
        active_tasks = 0;
        int task_index = 0;
        for (int j = 0; j < tracker_size; ++j) {
            // Detect task status changes
            if (active_tracker[j] == 1 && prev_active_tracker[j] == 0) {
                added_tasks.push_back(j);
            }else if (active_tracker[j] == 0 && prev_active_tracker[j] == 1) {
                removed_tasks.push_back(j);
            }
            // Update task_index 
            if (active_tracker[j] == 1) {
                // Choose the active task with lowest id as new root
                if (active_tasks == 0) {
                    if (mpi_root != j && id == j) {
                        std::cout << "UPDATE MPI_ROOT: " << id << " BECOME NEW ROOT" << std::endl;
                    }
                    mpi_root = j;
                }
                // The task new index is the its id ranked in the list of remaining active tasks
                if (j == id) {
                    task_index = active_tasks;
                }
                active_tasks++;
            }
        }

        bool restarted = false;
        if (active_tasks == 0) {
            if (allow_restart == false) {
                std::cerr << "All tasks failed, stop the reconstruction" << std::endl;
                return 1;
            }
            // All tasks are stopped, restart the computation from beginning
            MPI_Barrier(MPI_COMM_WORLD);
            ckpt->checkpoint_wait();
            
            mpi_root = 0;
            if (id == mpi_root) {
                std::cout << "ALL TASKS HAVE STOPPED. RESTART ALL THE RECONSTRUCTION TASKS" << std::endl;
            }
            task_is_active = 1;
            for (int i = 0; i < num_tasks; ++i) {
                active_tracker[i] = 1;
            }
            task_stop_threshold = exp_dist(gen);
            // std::vector<double> next_task_stop_thresholds = {186.601, 98.5418, 138.083, 97.1347};
            // task_stop_threshold = next_task_stop_thresholds[id];
            std::cout << "[Task" << id << "] will stop in " << task_stop_threshold << " second(s)." << std::endl;
            std::chrono::duration<double> recon_progress = std::chrono::high_resolution_clock::now() - recon_start;
            task_stop_threshold += recon_progress.count();
            num_rows = 0;
            local_active_index = 0;

            // Reload checkpoints
            if (progress > 0) {
                // progress--;
                std::cout << "[Task-" << id << "]: Recover checkpoint " << id << " from progress " << progress << std::endl;
                num_rows = dy;
                int v = progress-1;
                recover(ckpt, id, ckpt_name, sinogram_size, v, &num_ckpt, num_rows, local_recon, row_indexes, local_progress, true);
                for (int i = 0; i < num_rows; ++i) {
                    memcpy(local_data + i*dt*dx, data_swap+row_indexes[i]*dt*dx, sizeof(float)*dt*dx);
                }
                std::cout << "[task-" << id << "]: Recovery completed, checkpoint: " << id << ", version: " << v << " num_row: " << num_rows << " num_ckpt: " << num_ckpt << std::endl;
                MPI_Bcast(&num_ckpt, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
                progress--;
            }

            int * num_row_trackers = new int [num_tasks];
            MPI_Allgather(&num_rows, 1, MPI_INT, num_row_trackers, 1, MPI_INT, MPI_COMM_WORLD);
            added_tasks.clear();
            for (int i = 0; i < num_tasks; ++i) {
                if (num_row_trackers[i] == 0) {
                    added_tasks.push_back(i);
                }
            }
            removed_tasks.clear();
            active_tasks = num_tasks;
            task_index = id;
            delete [] num_row_trackers;
            restarted = true;

            // std::exit(1);

        }

        auto recovery_start = std::chrono::high_resolution_clock::now();

        if (!removed_tasks.empty()) {
            // Some tasks fails, recover their progress from checkpoints
            stat_num_failures += removed_tasks.size();
            if (id == mpi_root) {
                std::cout << "Found " << removed_tasks.size() << " task(s) stop working, start recovery..." << std::endl;
            }
            if (task_is_active) {
                // Remaining active tasks recover the checkpoints of inactive ones
                int num = (removed_tasks.size() / active_tasks);
                if (removed_tasks.size() % active_tasks > task_index) {
                    num++;
                }
                // Read data from checkpoints
                bool ckpt_loaded = false; 
                for (int j = 0; j < num; ++j) {
                    int ckpt_size = removed_tasks.size();
                    int numread = dy;
                    int v = progress-1;
                    // unsigned int ckpt_id = removed_tasks[removed_tasks.size()*j + task_index];
                    unsigned int ckpt_id = removed_tasks[active_tasks*j + task_index];
                    std::cout << "[Task-" << id << "]: Recover checkpoint " << ckpt_id << " from progress " << v << std::endl;
                    ckpt->checkpoint_wait();
                    recover(ckpt, ckpt_id, ckpt_name, sinogram_size, v, &ckpt_size, numread, local_recon+num_rows*sinogram_size, row_indexes+num_rows, local_progress+num_rows);
                    std::cout << "[Task-" << id << "]: Recovery completed, checkpoint: " << ckpt_id << ", progress: " << v << " num_row: " << numread << " num_ckpt: " << ckpt_size << std::endl;
                    // Update input data for tasks receiving new slices
                    for (int i = num_rows; i < num_rows+numread; ++i) {
                        std::cout << "[Task-" << id << "]: " << "Recovered row: at #" << row_indexes[i] << ", progress: " << local_progress[i] << std::endl;
                        memcpy(local_data + i*dt*dx, data_swap+row_indexes[i]*dt*dx, sizeof(float)*dt*dx);
                    }
                    num_rows += numread;
                    ckpt_loaded = true;
                }
                if (ckpt_loaded) {
                    ckpt = veloc::get_client((unsigned int)task_index, check_point_config);
                    ckpt->mem_protect(0, &num_ckpt, 1, sizeof(int));
                    ckpt->mem_protect(1, &num_rows, 1, sizeof(int));
                    ckpt->mem_protect(2, local_recon, std::max(num_rows, 1)*sinogram_size, sizeof(float));
                    ckpt->mem_protect(3, row_indexes, std::max(num_rows, 1), sizeof(int));
                    ckpt->mem_protect(4, local_progress, std::max(num_rows, 1), sizeof(int));
                }
            }else{
                num_rows = 0;
            }
        }

        // Reorganize reconstruction data: sorting rows by their progress
        if (task_is_active && num_rows > 0) {
            int *pindexes = new int[num_rows];
            for (int i = 0; i < num_rows; ++i) {
                pindexes[i] = i;
            }
            std::sort(pindexes, pindexes+num_rows, [local_progress](int i, int j) {
                return local_progress[i] > local_progress[j];
            });
            float *tmp_local_recon = new float[num_rows*sinogram_size];
            int *tmp_row_indexes = new int[num_rows];
            int *tmp_local_progress = new int[num_rows];
            memcpy(tmp_local_recon, local_recon, sizeof(float)*num_rows*sinogram_size);
            memcpy(tmp_row_indexes, row_indexes, sizeof(int)*num_rows);
            memcpy(tmp_local_progress, local_progress, sizeof(int)*num_rows);
            for (int i=0; i < num_rows; ++i) {
                row_indexes[i] = tmp_row_indexes[pindexes[i]];
                local_progress[i] = tmp_local_progress[pindexes[i]];
                memcpy(local_recon + i*sinogram_size, tmp_local_recon + pindexes[i]*sinogram_size, sizeof(float)*sinogram_size);
                memcpy(local_data + i*dt*dx, data_swap+row_indexes[i]*dt*dx, sizeof(float)*dt*dx);
            }
            delete []tmp_local_recon;
            delete []tmp_row_indexes;
            delete []tmp_local_progress;
        }

        // Calculate number of remaining rows/slices
        local_active_index = 0;
        if (task_is_active) {
            while (local_active_index < num_rows && local_progress[local_active_index] >= num_outer_iter) {
                local_active_index++;
            }
        }
        int collected_completed_row = 0;
        int ws = 0;
        MPI_Allreduce(&local_active_index, &collected_completed_row, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
        
        // Calculate number of rows/slices each task need to process in the next iteration
        if (collected_completed_row == dy) {
            // Complete
            std::string rindex = " [";
            for (int i = num_rows-1; i >= 0; --i) {
                rindex += " " + std::to_string(row_indexes[i]) + ":" + std::to_string(local_progress[i]);
            }
            rindex += "]";
            std::cout << "[Task-" << id << "]: Handling " << num_rows << " rows: " << num_rows << " ws: " << ws << rindex << std::endl;
            break;
        }else{
            ws = (dy - collected_completed_row) / active_tasks;
        }

        // Based on the assign computation, determine tasks that are overloaded/underloaded
        int loaddiff = 0;
        if (task_is_active) {
            loaddiff = (num_rows - local_active_index) - ws;
        }
        int * collected_loaddiff = new int [num_tasks];
        int * overloads = new int [num_tasks];
        MPI_Allgather(&loaddiff, 1, MPI_INT, collected_loaddiff, 1, MPI_INT, MPI_COMM_WORLD);
        int total_overload = 0;
        for (int i = 0; i < num_tasks; ++i) {
            overloads[i] = std::max(collected_loaddiff[i], 0);
            total_overload += overloads[i];
        }

        // Rebalance if needed
        if (total_overload > 0 && active_tasks > 1) {
            if (id == mpi_root) {
                std::cout << "[Task-" << id << "]: Load is unbalance. total_overload: " << total_overload << std::endl; 
            }
        // if (total_overload > num_tasks && active_tasks > 1) {
            // Load is imbalance, redistribute slices.
            float *tmp_recon = new float[recon_size];
            int *tmp_row_indexes = new int[dy];
            int *tmp_local_progress = new int[dy];
            int *displacements = new int[num_tasks];
            // Send overloaded slices to root
            // if (id == mpi_root) {
            //     std::cout << "[Task-" << id << "]: Load imbalance: " << total_overload << std::endl;
            // }
            displacements[0] = 0;
            for (int i = 1; i < num_tasks; ++i) {
                displacements[i] = displacements[i-1] + overloads[i-1];

            }
            MPI_Gatherv(row_indexes+num_rows-overloads[id], overloads[id], MPI_INT, tmp_row_indexes, overloads, displacements, MPI_INT, mpi_root, MPI_COMM_WORLD);
            MPI_Gatherv(local_progress+num_rows-overloads[id], overloads[id], MPI_INT, tmp_local_progress, overloads, displacements, MPI_INT, mpi_root, MPI_COMM_WORLD);
            for (int i = 0; i < num_tasks; ++i) {
                displacements[i] *= sinogram_size;
                overloads[i] *= sinogram_size;
            }
            MPI_Gatherv(local_recon+num_rows*sinogram_size-overloads[id], overloads[id], MPI_FLOAT, tmp_recon, overloads, displacements, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
            num_rows -= overloads[id] / sinogram_size;

            // Determine number of rows each task would receive
            int *transferred_rows = new int [num_tasks];
            for (int i = 0; i < num_tasks; ++i) {
                transferred_rows[i] = 0;
            }
            
            if (id == mpi_root) {
                int *ld_indexes = new int [num_tasks];
                for (int i = 0; i < num_tasks; ++i) {
                    ld_indexes[i] = i;
                }
                std::sort(ld_indexes, ld_indexes+num_tasks, [collected_loaddiff](int i, int j) {
                    return collected_loaddiff[i] < collected_loaddiff[j];
                });
                std::cout << "[Task-" << id << "]: Trying to move data from overloaded tasks to underloaded onces" << std::endl; 
                // for (int i = 0; i < num_tasks; ++i) {
                //     std::cout << "ld_indexes " << i << " " << ld_indexes[i] << " -> " << collected_loaddiff[ld_indexes[i]] << std::endl;
                // }
                int i = 0;
                int j = num_tasks-1;
                int remain_load = total_overload;
                // Move slices to underloaded tasks if any
                // std::cout << "Shifting slices from overload to underload tasks" << std::endl;
                while (i < j && collected_loaddiff[ld_indexes[i]] < 0) {
                    if (-collected_loaddiff[ld_indexes[i]] > collected_loaddiff[ld_indexes[j]]) {
                        transferred_rows[ld_indexes[i]] += collected_loaddiff[ld_indexes[j]];
                        collected_loaddiff[ld_indexes[i]] += collected_loaddiff[ld_indexes[j]];
                        remain_load -= collected_loaddiff[ld_indexes[j]];
                        collected_loaddiff[ld_indexes[j]] = 0;
                        j--;
                    }else{
                        transferred_rows[ld_indexes[i]] += -collected_loaddiff[ld_indexes[i]];
                        collected_loaddiff[ld_indexes[j]] += collected_loaddiff[ld_indexes[i]];
                        remain_load -= -collected_loaddiff[ld_indexes[i]];
                        collected_loaddiff[ld_indexes[i]] = 0;
                        i++;
                        if (collected_loaddiff[ld_indexes[j]] == 0) {
                            j--;
                        }
                    }
                }
                // std::cout << "Shifting slices to normal tasks: " << i << " " << j  << std::endl;
                // Shift slices among tasks if there are still overloaded tasks
                // int k = 0;
                // int transfer_limit = (remain_load-1) / active_tasks + 1;
                int transfer_limit = (remain_load-1) / active_tasks + 1;
                std::cout << "[Task-" << id << "]: Remain total_overload: " << remain_load << ". Spread them among tasks. Dynamic Redistribution transfer_limit: " << transfer_limit << std::endl;
                // std::cout << "[Task-" << id << "]: i = " << i << " j = " << j << " ld_indexes[j] = " << ld_indexes[j] << " collected_loaddiff = " << collected_loaddiff[ld_indexes[j]] << std::endl;
                while (j >= 0 && collected_loaddiff[ld_indexes[j]] > 0) {
                    // std::cout << "[Task-" << id << "]: OUTER -- i = " << i << "j = " << j << "ld_indexes[j] = " << ld_indexes[j] << " collected_loaddiff = " << collected_loaddiff[ld_indexes[j]] << std::endl;
                    while (collected_loaddiff[ld_indexes[j]] > 0) {
                        // std::cout << "[Task-" << id << "]: INNER -- i = " << i << "j = " << j << "ld_indexes[j] = " << ld_indexes[j] << " collected_loaddiff = " << collected_loaddiff[ld_indexes[j]] << std::endl;
                        int k = (ld_indexes[j]+1) % num_tasks;
                        // do {
                        //     k = (k+1) % num_tasks;
                        // } while (active_tracker[k] != 1);
                        int numtry = 0;
                        while (active_tracker[k] != 1 || transferred_rows[k] >= transfer_limit) {
                            k = (k+1) % num_tasks;
                            numtry++;
                            // if we all tasks meet the transfer limit yet there are data to be moved,
                            // then increase the transfer limit
                            if (numtry > num_tasks) {
                                transfer_limit++;
                            }
                        }
                        std::cout << "Tranfer 1 row from Task #" << ld_indexes[j] << " to Task #" << k << std::endl;
                        transferred_rows[k]++;
                        collected_loaddiff[ld_indexes[j]]--;
                        remain_load--;
                    }
                    j--;
                }
                delete[] ld_indexes;
            }
            // std::cout << "[Task-" << id << "]: overload: " << loaddiff << std::endl;
            MPI_Bcast(transferred_rows, num_tasks, MPI_INT, mpi_root, MPI_COMM_WORLD);
            displacements[0] = 0;
            for (int i = 1; i < num_tasks; ++i) {
                displacements[i] = displacements[i-1] + transferred_rows[i-1];
            }
            
            MPI_Scatterv(tmp_row_indexes, transferred_rows, displacements, MPI_INT, row_indexes+num_rows, transferred_rows[id], MPI_INT, mpi_root, MPI_COMM_WORLD);
            MPI_Scatterv(tmp_local_progress, transferred_rows, displacements, MPI_INT, local_progress+num_rows, transferred_rows[id], MPI_INT, mpi_root, MPI_COMM_WORLD);
            for (int i = 0; i < num_tasks; ++i) {
                displacements[i] *= sinogram_size;
                transferred_rows[i] *= sinogram_size;
            }
            MPI_Scatterv(tmp_recon, transferred_rows, displacements, MPI_FLOAT, local_recon + num_rows*sinogram_size, transferred_rows[id], MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
            for (int i = 0; i < num_tasks; ++i) {
                transferred_rows[i] /= sinogram_size;
            }
            // std::cout << "Task-" << id << "]: transferred_row: " << transferred_rows[id] << std::endl;

            // Update input data for tasks receiving new slices
            if (task_is_active) {
                for (int i = 0; i < transferred_rows[id]; ++i) {
                    memcpy(local_data + num_rows*dt*dx + i*dt*dx, data_swap+row_indexes[num_rows+i]*dt*dx, sizeof(float)*dt*dx);
                }
            }
            num_rows += transferred_rows[id];

            if (task_is_active) {
                ckpt = veloc::get_client((unsigned int)task_index, check_point_config);
                ckpt->mem_protect(0, &num_ckpt, 1, sizeof(int));
                ckpt->mem_protect(1, &num_rows, 1, sizeof(int));
                ckpt->mem_protect(2, local_recon, std::max(num_rows, 1)*sinogram_size, sizeof(float));
                ckpt->mem_protect(3, row_indexes, std::max(num_rows, 1), sizeof(int));
                ckpt->mem_protect(4, local_progress, std::max(num_rows, 1), sizeof(int));
            }

            delete[] tmp_recon;
            delete[] tmp_row_indexes;
            delete[] tmp_local_progress;
            delete[] displacements;

            delete[] transferred_rows;
        }
        delete[] collected_loaddiff;
        delete[] overloads;

        // Adjust number of slices to be computed
        if (ws == 0) {
            ws = num_rows - local_active_index;
        }else{
            ws = std::min(ws, num_rows - local_active_index);
        }
        if (task_is_active) {
            std::string rindex = " [";
            if (ws == 0) {
                rindex += " | ";
            }
            for (int i = num_rows-1; i >= 0; --i) {
                rindex += " " + std::to_string(row_indexes[i]) + ":" + std::to_string(local_progress[i]);
                if (ws == num_rows-i) {
                    rindex += " | ";
                }else{
                    rindex += ", ";
                }
            }
            rindex += "]";
            std::cout << "[Task-" << id << "]: Handling " << num_rows << " rows: " << num_rows << " ws: " << ws << rindex << std::endl;
        }

        auto recovery_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> recovery_elapsed_time = recovery_end - recovery_start;
        recovery_time += recovery_elapsed_time.count();

        auto comm_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> comm_elapsed_time = comm_end - comm_start;
        comm_time += comm_elapsed_time.count();

        // Save progress with checkpoint
        num_ckpt = active_tasks;
        auto ckpt_start = std::chrono::high_resolution_clock::now();
        if (task_is_active && !restarted) {
            ckpt->checkpoint_wait();
            if (!ckpt->checkpoint(ckpt_name, progress)) {
                std::cout << "[Task-" << id << "] cannot checkpoint: numrow: " << num_rows << " progress " << progress << std::endl;
                throw std::runtime_error("Checkpointing failured");
            }
            std::cout << "[task-" << id << "]: Checkpointed version " << progress << std::endl;
        }

        auto ckpt_end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> ckpt_elapsed_time = ckpt_end - ckpt_start;
        ckpt_time += ckpt_elapsed_time.count();
        
        exec_start = std::chrono::high_resolution_clock::now();

        // Do the reconstruction
        if (task_is_active && ws > 0) {
            std::cout << "[Task-" << id << "]: Outer iteration: " << progress << std::endl;
            // art(data_swap, w_dy, w_dt, w_dx, &center, theta, w_recon, w_ngridx, w_ngridy, num_iter);
            // art(local_data, num_rows, dt, dx, &center, theta, local_recon, ngridx, ngridy, num_iter);
            float * ws_data = local_data + (num_rows - ws)*dt*dx;
            float * ws_local_recon = local_recon + (num_rows-ws)*sinogram_size;
            // art(ws_data, ws, dt, dx, &center, theta, ws_local_recon, ngridx, ngridy, num_iter);
            
            auto current_time = std::chrono::high_resolution_clock::now();
            std::chrono::duration<double> used_time = (current_time - recon_start);
            double remain_time = task_stop_threshold - used_time.count();
            // std::cout << "[Task-" << id << "]: Remain time = " << remain_time << std::endl;
            if (remain_time / ws < reconstruction_time_per_iter) {
                // We will crash during reconstruction, try sleep instead
                std::cout << "[Task-" << id << "]: Remain time: " << remain_time << "/" << ws << " = " << remain_time/ws << " < " << reconstruction_time_per_iter << " DONT HAVE ENOUGH TIME TO RECONSTRUCT, SLEEP INTEAD" << std::endl;
                std::this_thread::sleep_for(std::chrono::seconds(std::max(1, int(remain_time+1))));
                found_crashed = true;
            }else{
                auto iter_start = std::chrono::high_resolution_clock::now();
                recon_simple(recon_method, ws_data, ws, dt, dx, &center, theta, ws_local_recon, ngridx, ngridy, num_iter);
                auto iter_end = std::chrono::high_resolution_clock::now();
                std::chrono::duration<double> iter_time = (iter_end - iter_start);
                reconstruction_time_per_iter = iter_time.count() / ws;
                for (int i = num_rows-ws; i < num_rows; ++i) {
                    local_progress[i]++; 
                }
                found_crashed = false;
            }
        }

        // Make sure the partial progress is saved in case some task failed right after the restart
        if (restarted) {
            ckpt->checkpoint_wait();
            if (!ckpt->checkpoint(ckpt_name, progress)) {
                std::cout << "[Task-" << id << "] cannot checkpoint: numrow: " << num_rows << " progress " << progress << std::endl;
                throw std::runtime_error("Checkpointing failured");
            }
            std::cout << "[task-" << id << "]: Checkpointed version " << progress << std::endl;
        }

        progress++;

        elapsed_time = std::chrono::high_resolution_clock::now() - recon_start;
        // if (!restarted && (elapsed_time.count() > task_stop_threshold || found_crashed) && (allow_restart || id != mpi_root)) {
        if ((elapsed_time.count() > task_stop_threshold || found_crashed) && (allow_restart || id != mpi_root)) {
            if (task_is_active) {
                std::cout << "WARNING: Task " << id << " has stopped." << std::endl;
            }
            task_is_active = 0;
        }else{
            task_is_active = 1;
        }

        // if (progress == 0 && id > 8) {
        //     task_is_active = 0;
        // }
        // if (progress == 7 && id > 8) {
        //     task_is_active = 0;
        // }
        task_state_history.push_back(task_is_active);

    }

    std::cout << "[Task " << id << "] Complete the reconstruction, waiting for other..." << std::endl;

    std::string state_info = "[";
    for (int i = 0; i < task_state_history.size(); ++i) {
        state_info += " " + std::to_string(i) + ": " + std::to_string(task_state_history[i]) + ", ";
    }
    state_info += "]";
    std::cout << "[Task-" << id << "]: State history: " << state_info << std::endl;
    
    auto recon_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> recon_elapsed = recon_end - recon_start;
    double recon_time = recon_elapsed.count();

    // const char * img_name = "recon.h5";
    const char * img_name = "recon";
    if (id == mpi_root) {
        std::cout << "Reconstructed data from workers" << std::endl;
    }
    float * tmp_recon = nullptr;
    int * tmp_row_indexes = nullptr;
    int * collected_rows = nullptr;
    int * displacements = nullptr;
    if (id == mpi_root) {
        tmp_recon = new float[recon_size];
        tmp_row_indexes = new int [dy];
        collected_rows = new int[num_tasks];
        displacements = new int[num_tasks];
    } 
    MPI_Gather(&num_rows, 1, MPI_INT, collected_rows, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
    if (id == mpi_root) {
        displacements[0] = 0;
        for (int i = 1; i < num_tasks; ++i) {
            displacements[i] = displacements[i-1] + collected_rows[i-1];
        }
    }
    MPI_Gatherv(row_indexes, num_rows, MPI_INT, tmp_row_indexes, collected_rows, displacements, MPI_INT, mpi_root, MPI_COMM_WORLD);
    if (id == mpi_root) {
        for (int i = 0; i < num_tasks; ++i) {
            displacements[i] *= sinogram_size;
            collected_rows[i] *= sinogram_size;
        }
    }
    MPI_Gatherv(local_recon, num_rows*sinogram_size, MPI_FLOAT, tmp_recon, collected_rows, displacements, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);

    // MPI_Gather(local_recon, num_rows*sinogram_size, MPI_FLOAT, tmp_recon, num_rows*sinogram_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
    // MPI_Gather(row_indexes, num_rows, MPI_INT, tmp_row_indexes, num_rows, MPI_INT, mpi_root, MPI_COMM_WORLD);
    if (id == mpi_root) {
        for (int i = 0; i < dy; ++i) {
            std::cout << "Write data " << tmp_row_indexes[i] << " <-- " << i << std::endl;
            memcpy(recon + tmp_row_indexes[i]*sinogram_size, tmp_recon + i*sinogram_size, sizeof(float)*sinogram_size);
        }
        delete [] tmp_recon;
        delete [] tmp_row_indexes;
        delete [] collected_rows;
        delete [] displacements;
    
        // write the reconstructed data to a file
        // Create the output file name
        std::ostringstream oss;
        oss << img_name << "-" << recon_method << ".h5";
        std::string output_filename = oss.str();
        const char* output_filename_cstr = output_filename.c_str();

        hsize_t output_dims[3] = {dy, ngridy, ngridx};
        if (saveAsHDF5(output_filename_cstr, recon, output_dims) != 0) {
            std::cerr << "Error: Unable to create file " << output_filename << std::endl;
            return 1;
        }
        else{
            std::cout << "Save the reconstruction image as " << oss.str() << std::endl;
        }

    }

    if (id == mpi_root) {
        // Dump the reconsruction configuration and timing to a file
        std::string filePath = "execinfo.json";
        std::ofstream ofile;
        ofile.open(filePath, std::ios::app);

        if (ofile.is_open()) {
            ofile << "{" << std::endl;
            ofile << "\"prob\" : " << failure_prob << "," << std::endl;
            ofile << "\"nprocs\" : " << num_tasks << "," << std::endl;
            ofile << "\"nslices\" : " << nslices << "," << std::endl;
            ofile << "\"recon_method\" : \"" << recon_method << "\"," << std::endl;
            ofile << "\"approach\" : " << "\"ckpt-dynamic-redis\"" << "," << std::endl;
            ofile << "\"num_iter\" : " << num_outer_iter*num_iter << "," << std::endl;
            ofile << "\"allow_restart\" : " << allow_restart << "," << std::endl;
            ofile << "\"filename\" : \"" << filename << "\"," << std::endl;
            ofile << "\"ngridx\" : " << ngridx << "," << std::endl;
            ofile << "\"ngridy\" : " << ngridy << "," << std::endl;
            ofile << "\"theta\" : " << dt << "," << std::endl;
            ofile << "\"total\" : " << recon_time << "," << std::endl;
            ofile << "\"exec\" : " << exec_time << "," << std::endl;
            ofile << "\"ckpt\" : " << ckpt_time << "," << std::endl;
            ofile << "\"comm\" : " << comm_time << "," << std::endl;
            ofile << "\"recover\" : " << recovery_time << "," << std::endl;
            ofile << "\"task_failures\" : " << stat_num_failures << std::endl;
            ofile << "}," << std::endl;
        }else{
            std::cerr << "Cannot save the experiment configuration and timing" << std::endl;
        }

        ofile.close();

        std::cout << "ELAPSED TIME: " << recon_time << " seconds" << std::endl;
        std::cout << "EXEC TIME: " << exec_time << " seconds" << std::endl;
        std::cout << "CHECKPOINTING TIME: " << ckpt_time << " seconds" << std::endl;
        std::cout << "COMM TIME: " << comm_time << " seconds" << std::endl;
        std::cout << "RECOVERY TIME: " << recovery_time << " seconds" << std::endl;
    }


    // free the memory
    delete[] data;
    delete[] data_swap;
    delete[] theta;
    delete[] recon;
    delete[] local_recon;
    delete[] row_indexes;
    delete[] local_progress;
    delete[] local_data;

    MPI_Finalize();

    return 0;
}
