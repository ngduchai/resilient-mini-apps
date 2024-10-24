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
#include <random>
#include <chrono>
#include <fstream>

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

void recover(veloc::client_t *ckpt, int id, const char *name,  int sinogram_size, int &progress, int *num_ckpt, int &numrows, float *recon, int *row_index, bool exact_ver=false) {
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
    // if (ckpt->restart(name, progress, id) == false) {
    //     numrows = 0;
    // }

}

int main(int argc, char* argv[])
{

    if(argc != 10) {
        std::cerr << "Usage: " << argv[0] << " <filename> <center> <num_outer_iter> <num_iter> <beginning_sino> <num_sino> <failure_prob> <allow_restart> [veloc config]" << std::endl;
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
    const char* check_point_config = (argc == 10) ? argv[9] : "art_simple.cfg";

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
    unsigned int mpi_root = 0;
    
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

    // Setup random number generation
    std::random_device rd;  // Seed generator
    std::mt19937 gen(rd()); // Mersenne Twister engine
    std::exponential_distribution exp_dist(failure_prob); // Distribution for 0 and 1
    double task_stop_threshold = exp_dist(gen);
    std::cout << "[Task-" << id << "] will stop in the next " << task_stop_threshold << " second(s)." << std::endl;

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
    
    if (id == mpi_root) {
        progress = ckpt->restart_test(ckpt_name, 0, id);
    }
    MPI_Bcast(&progress, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
    
    // Load the checkpoint if any 
    if (progress > 0) {
        std::cout << "[Task-" << id << "]: Recover checkpoint " << id << " from progress " << progress << std::endl;
        num_rows = dy;
        recover(ckpt, id, ckpt_name, sinogram_size, progress, &num_ckpt, num_rows, local_recon, row_indexes);
        for (int i = 0; i < num_rows; ++i) {
            memcpy(local_data + i*dt*dx, data_swap+row_indexes[i]*dt*dx, sizeof(float)*dt*dx);
        }
        std::cout << "[task-" << id << "]: Recovery completed, checkpoint: " << id << ", progress: " << progress << " num_row: " << num_rows << " num_ckpt: " << num_ckpt << std::endl;
        MPI_Bcast(&num_ckpt, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
    }else{
        progress = 0;
        ckpt->mem_protect(1, &num_rows, 1, sizeof(int));
        ckpt->mem_protect(2, local_recon, num_rows*sinogram_size, sizeof(float));
        ckpt->mem_protect(3, row_indexes, num_rows, sizeof(int));
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

    double ckpt_time = 0;
    double recovery_time = 0;
    double comm_time = 0;
    auto recon_start = std::chrono::high_resolution_clock::now();

    // run the reconstruction
    while (progress < num_outer_iter+1) {
        // We need one extra iteration to ensure all rows are sync with the expected number of iterations
        
        // MPI_Barrier(MPI_COMM_WORLD);

        auto comm_start = std::chrono::high_resolution_clock::now();

        // Sync the task status across all tasks
        std::swap(active_tracker, prev_active_tracker);
        MPI_Allgather(&task_is_active, 1, MPI_INT, active_tracker, 1, MPI_INT, MPI_COMM_WORLD);
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
        if (active_tasks == 0  && allow_restart) {
            // All tasks are stopped, restart the computation from beginning
            MPI_Barrier(MPI_COMM_WORLD);
            ckpt->checkpoint_wait();
            
            mpi_root = 0;
            if (id == mpi_root) {
                std::cout << "ALL TASKS HAVE STOPPED. RESTART ALL THE RECONSTRUCTION TASKS" << std::endl;
            }
            task_is_active = 1;
            task_stop_threshold = exp_dist(gen);
            std::cout << "[Task" << id << "] will stop in " << task_stop_threshold << " second(s)." << std::endl;
            std::chrono::duration<double> recon_progress = std::chrono::high_resolution_clock::now() - recon_start;
            task_stop_threshold += recon_progress.count();
            num_rows = 0;

            // Reload checkpoints
            if (progress > 0) {
                std::cout << "[Task-" << id << "]: Recover checkpoint " << id << " from progress " << progress << std::endl;
                num_rows = dy;
                int v = progress-1;
                recover(ckpt, id, ckpt_name, sinogram_size, v, &num_ckpt, num_rows, local_recon, row_indexes, true);
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
            delete [] num_row_trackers;
            restarted = true;
        }

        auto recovery_start = std::chrono::high_resolution_clock::now();

        if (!added_tasks.empty()) {
            if (id == mpi_root) {
                std::cout << "Found " << added_tasks.size() << " new task(s), start redistribution..." << std::endl;
            }
            // Some tasks are added, rebalance by moving some slices to new tasks
            int transferred_rows = 0;
            int adj_num_rows = 0;
            // First, determine if a task is just added and its index
            int transfer_index = -1;
            for (int i = 0; i < added_tasks.size(); ++i) {
                if (added_tasks[i] == id) {
                    transfer_index = i;
                    break;
                }
            }
            int total_rows = 0;
            int active_rows = 0;
            if (task_is_active) {
                active_rows = num_rows;
            }
            MPI_Allreduce(&active_rows, &total_rows, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            // Old task detemine the number of rows they will send to new tasks
            if (task_is_active && transfer_index == -1) {
                adj_num_rows = total_rows / active_tasks;
                int extra_rows = total_rows % active_tasks;
                // int w_offset = adj_num_rows*task_index + std::min(id, extra_rows);
                if (extra_rows != 0 && task_index < extra_rows) {
                    adj_num_rows++;
                }
                transferred_rows = std::max(num_rows - adj_num_rows, 0);
            }
            // std::cout << "[task-" << id << "] task_index: " << task_index << " transferred_rows: " << transferred_rows << std::endl; 
            // Transfer rows from old tasks to mpi_root
            float * collected_recon = nullptr;
            int * collected_indexes = nullptr;
            int * collected_transferred_rows = nullptr;
            int * displacements = nullptr;
            if (id == mpi_root) {
                collected_recon = new float[recon_size];
                collected_indexes = new int[dy];
                collected_transferred_rows = new int[num_tasks];
                displacements = new int[num_tasks];
            }
            int total_transferred_rows;
            MPI_Allreduce(&transferred_rows, &total_transferred_rows, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Gather(&transferred_rows, 1, MPI_INT, collected_transferred_rows, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
            if (id == mpi_root) {
                displacements[0] = 0;
                for (int i = 1; i < num_tasks; ++i) {
                    displacements[i] = displacements[i-1] + collected_transferred_rows[i-1];

                }
            }
            MPI_Gatherv(row_indexes+adj_num_rows, transferred_rows, MPI_INT, collected_indexes, collected_transferred_rows, displacements, MPI_INT, mpi_root, MPI_COMM_WORLD);
            if (id == mpi_root) {
                for (int i = 0; i < num_tasks; ++i) {
                    displacements[i] *= sinogram_size;
                    collected_transferred_rows[i] *= sinogram_size;
                }
            }
            MPI_Gatherv(local_recon+adj_num_rows*sinogram_size, transferred_rows*sinogram_size, MPI_FLOAT, collected_recon, collected_transferred_rows, displacements, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
            
            // std::cout << "[task-" << id << "] total_transferred_rows: " << total_transferred_rows << std::endl; 

            // Update the number of rows each that will process
            num_rows -= transferred_rows;

            // std::cout << "[Task-" << id << "]: Complete gathering rows at root" << std::endl;

            // Transfer the rows from old tasks to new tasks
            transferred_rows = 0;
            // if (task_is_active && transfer_index != -1) {
            //     // only receive data if the task was added
            //     transferred_rows = total_transferred_rows / added_tasks.size();
            //     extra_rows = total_transferred_rows % added_tasks.size();
            //     if (transfer_index < extra_rows) {
            //         transferred_rows++;
            //     }
            // }
            if (task_is_active) {
                adj_num_rows = total_rows / active_tasks;
                int extra_rows = total_rows % active_tasks;
                if (extra_rows != 0 && task_index < extra_rows) {
                    adj_num_rows++;
                }
                transferred_rows = std::max(adj_num_rows - num_rows, 0);
            }

            // std::cout << "[Task " << id << "]: data to be moved from other: " << transferred_rows << " total_rows: " << total_rows << " active_tasks: " << active_tasks << std::endl;

            MPI_Gather(&transferred_rows, 1, MPI_INT, collected_transferred_rows, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
            if (id == mpi_root) {
                displacements[0] = 0;
                for (int i = 1; i < num_tasks; ++i) {
                    displacements[i] = displacements[i-1] + collected_transferred_rows[i-1]; 
                }
            }
            MPI_Scatterv(collected_indexes, collected_transferred_rows, displacements, MPI_INT, row_indexes+num_rows, transferred_rows, MPI_INT, mpi_root, MPI_COMM_WORLD);
            if (id == mpi_root) {
                for (int i = 0; i < num_tasks; ++i) {
                    displacements[i] *= sinogram_size;
                    collected_transferred_rows[i] *= sinogram_size;
                }
            }
            MPI_Scatterv(collected_recon, collected_transferred_rows, displacements, MPI_FLOAT, local_recon + num_rows*sinogram_size, transferred_rows*sinogram_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
            

            // Update input data for tasks receiving new slices
            if (task_is_active && transfer_index != -1) {
                for (int i = 0; i < transferred_rows; ++i) {
                    memcpy(local_data + num_rows*dt*dx + i*dt*dx, data_swap+row_indexes[num_rows+i]*dt*dx, sizeof(float)*dt*dx);
                }
            }
            num_rows += transferred_rows;

            if (task_is_active) {
                std::string rindex = " [";
                for (int i = 0; i < num_rows; ++i) {
                    rindex += " " + std::to_string(row_indexes[i]) + ", ";
                }
                rindex += "]";
                std::cout << "[Task-" << id << "]: Complete collecting rows from root, new rows = " << transferred_rows << " -- total: " << num_rows <<  rindex << std::endl;
            }

            if (task_is_active) {
                ckpt = veloc::get_client((unsigned int)task_index, check_point_config);
                ckpt->mem_protect(0, &num_ckpt, 1, sizeof(int));
                ckpt->mem_protect(1, &num_rows, 1, sizeof(int));
                ckpt->mem_protect(2, local_recon, std::max(num_rows, 1)*sinogram_size, sizeof(float));
                ckpt->mem_protect(3, row_indexes, std::max(num_rows, 1), sizeof(int));
            }

            if (id == mpi_root) {
                delete [] collected_recon;
                delete [] collected_indexes;
                delete[] collected_transferred_rows;
                delete[] displacements;
            }
        }

        if (!removed_tasks.empty()) {
            // Some tasks fails, recover their progress from checkpoints
            if (id == mpi_root) {
                std::cout << "Found " << removed_tasks.size() << " task(s) stop working, start recovery..." << std::endl;
            }
            int recovered_size = 0;
            float * recovered_recon = nullptr;
            int * recovered_row_indexes = nullptr;
            int * ckpt_progress = nullptr;
            int * collected_recovered_rows = nullptr;
            int * displacements = nullptr;
            float * local_recovered_recon = new float [recon_size];
            int * local_recovered_row_indexes = new int [dy];
            int * local_ckpt_progress = new int [dy];
            if (task_is_active) {
                // Remaining active tasks recover the checkpoints of inactive ones
                int num = (removed_tasks.size() / active_tasks);
                if (removed_tasks.size() % active_tasks > task_index) {
                    num++;
                }
                // Read data from checkpoints
                for (int j = 0; j < num; ++j) {
                    int ckpt_size = removed_tasks.size();
                    int numread = dy;
                    int v = progress;
                    // unsigned int ckpt_id = removed_tasks[removed_tasks.size()*j + task_index];
                    unsigned int ckpt_id = removed_tasks[active_tasks*j + task_index];
                    std::cout << "[Task-" << id << "]: Recover checkpoint " << ckpt_id << " from progress " << v << std::endl;
                    recover(ckpt, ckpt_id, ckpt_name, sinogram_size, v, &ckpt_size, numread, local_recovered_recon+recovered_size*sinogram_size, local_recovered_row_indexes+recovered_size);
                    for (int k = 0; k < numread; ++k) {
                        local_ckpt_progress[recovered_size+k] = v;
                    }
                    std::cout << "[task-" << id << "]: Recovery completed, checkpoint: " << ckpt_id << ", progress: " << v << " num_row: " << numread << " num_ckpt: " << ckpt_size << std::endl;
                    recovered_size += numread;
                }
            }

            // Gather the checkpoints at the root then redistribute across tasks
            if (id == mpi_root) {
                recovered_recon = new float [recon_size];
                recovered_row_indexes = new int [dy];
                ckpt_progress = new int [dy];
                collected_recovered_rows = new int[num_tasks];
                displacements = new int[num_tasks];
            }

            int total_recovered_size = 0;
            MPI_Allreduce(&recovered_size, &total_recovered_size, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
            MPI_Gather(&recovered_size, 1, MPI_INT, collected_recovered_rows, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
            if (id == mpi_root) {
                displacements[0] = 0;
                for (int i = 1; i < num_tasks; ++i) {
                    displacements[i] = displacements[i-1] + collected_recovered_rows[i-1];

                }
            }
            // Sync with root
            MPI_Gatherv(local_recovered_row_indexes, recovered_size, MPI_INT, recovered_row_indexes, collected_recovered_rows, displacements, MPI_INT, mpi_root, MPI_COMM_WORLD);
            MPI_Gatherv(local_ckpt_progress, recovered_size, MPI_INT, ckpt_progress, collected_recovered_rows, displacements, MPI_INT, mpi_root, MPI_COMM_WORLD);
            if (id == mpi_root) {
                for (int i = 0; i < num_tasks; ++i) {
                    displacements[i] *= sinogram_size;
                    collected_recovered_rows[i] *= sinogram_size;
                }
            }
            MPI_Gatherv(local_recovered_recon, recovered_size*sinogram_size, MPI_FLOAT, recovered_recon, collected_recovered_rows, displacements, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);

            // if (id == mpi_root) {
            //     std::cout << "[task-" << id << "]: complete collecting data at root" << std::endl;
            // }

            // Estimate the data size each task will receive
            if (task_is_active) {
                // recovered_size = total_recovered_size / active_tasks;
                // if (total_recovered_size % active_tasks > active_tasks - 1 - task_index) {
                // // if (total_recovered_size % active_tasks > task_index) {
                //     recovered_size++;
                // }
                int adj_num_rows = dy / active_tasks;
                if (dy % active_tasks > task_index) {
                // if (total_recovered_size % active_tasks > task_index) {
                    adj_num_rows++;
                }
                recovered_size = std::max(adj_num_rows - num_rows, 0);
            }else{
                recovered_size = 0;
            }

            MPI_Gather(&recovered_size, 1, MPI_INT, collected_recovered_rows, 1, MPI_INT, mpi_root, MPI_COMM_WORLD);
            if (id == mpi_root) {
                displacements[0] = 0;
                for (int i = 1; i < num_tasks; ++i) {
                    displacements[i] = displacements[i-1] + collected_recovered_rows[i-1];
                }
            }
            
            // Reditribute data from root
            MPI_Scatterv(recovered_row_indexes, collected_recovered_rows, displacements, MPI_FLOAT, local_recovered_row_indexes, recovered_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
            MPI_Scatterv(ckpt_progress, collected_recovered_rows, displacements, MPI_FLOAT, local_ckpt_progress, recovered_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);
            if (id == mpi_root) {
                for (int i = 0; i < num_tasks; ++i) {
                    displacements[i] *= sinogram_size;
                    collected_recovered_rows[i] *= sinogram_size;
                }
            }
            MPI_Scatterv(recovered_recon, collected_recovered_rows, displacements, MPI_FLOAT, local_recovered_recon, recovered_size*sinogram_size, MPI_FLOAT, mpi_root, MPI_COMM_WORLD);

            // Make sure the rows are up to date
            if (task_is_active && recovered_size > 0) {
                float * tmp_recon = new float [recovered_size*sinogram_size];
                int * tmp_indexes = new int [recovered_size];
                float * tmp_data = new float[recovered_size*dx*dt];
                int *tmp_progress = new int[recovered_size];

                while (recovered_size > 0) {
                    std::cout << "[task-" << id << "]: Sync " << recovered_size << " rows from checkpoints" << std::endl;
                    int remain_size = 0;
                    for (int i = 0; i < recovered_size; ++i) {
                        if (local_ckpt_progress[i] == progress) {
                            memcpy(local_recon + num_rows*sinogram_size, local_recovered_recon + i*sinogram_size, sizeof(float)*sinogram_size);
                            row_indexes[num_rows] = local_recovered_row_indexes[i];
                            memcpy(local_data + num_rows*dt*dx, data_swap+local_recovered_row_indexes[i]*dt*dx, sizeof(float)*dt*dx);
                            num_rows++;
                        }else{
                            std::cout << "Row #" << local_recovered_row_indexes[i] << " need to be sync" << std::endl;
                            memcpy(tmp_recon + remain_size*sinogram_size, local_recovered_recon + i*sinogram_size, sizeof(float)*sinogram_size);
                            tmp_indexes[remain_size] = local_recovered_row_indexes[i];
                            memcpy(tmp_data + remain_size*dt*dx, data_swap+local_recovered_row_indexes[i]*dt*dx, sizeof(float)*dt*dx);
                            tmp_progress[remain_size] = local_ckpt_progress[i];
                            remain_size++;
                        }
                    }
                    // std::cout << "[Task " << id << "]: updating " << remain_size << " rows" << std::endl;
                    std::swap(tmp_recon, local_recovered_recon);
                    std::swap(tmp_indexes, local_recovered_row_indexes);
                    std::swap(tmp_progress, local_ckpt_progress);
                    recovered_size = remain_size;
                    // std::cout << "Recover size: " << recovered_size << std::endl;
                    // Reconstruct from checkpoints
                    if (recovered_size > 0) {
                        art(tmp_data, recovered_size, dt, dx, &center, theta, local_recovered_recon, ngridx, ngridy, num_iter);
                        for (int i=0; i < recovered_size; ++i) {
                            local_ckpt_progress[i]++;
                        }
                    }
                    
                }
                delete [] tmp_recon;
                delete [] tmp_indexes;
                delete [] tmp_data;
                delete [] tmp_progress;
            }
            if (!task_is_active) {
                num_rows = 0;
            }

            if (task_is_active) {
                std::string rindex = " [";
                for (int i = 0; i < num_rows; ++i) {
                    rindex += " " + std::to_string(row_indexes[i]) + ", ";
                }
                rindex += "]";
                std::cout << "[Task-" << id << "]: Complete collecting rows from root, new rows = " << recovered_size << " -- total: " << num_rows <<  rindex << std::endl;
            }
            
            if (task_is_active) {
                ckpt = veloc::get_client((unsigned int)task_index, check_point_config);
                ckpt->mem_protect(0, &num_ckpt, 1, sizeof(int));
                ckpt->mem_protect(1, &num_rows, 1, sizeof(int));
                ckpt->mem_protect(2, local_recon, std::max(num_rows, 1)*sinogram_size, sizeof(float));
                ckpt->mem_protect(3, row_indexes, std::max(num_rows, 1), sizeof(int));
            }

            delete [] local_recovered_recon;
            delete [] local_recovered_row_indexes;
            delete [] local_ckpt_progress;
            if (id == mpi_root) {
                delete [] recovered_recon;
                delete [] recovered_row_indexes;
                delete [] ckpt_progress;
            }

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

        // Update state
        // if (progress < task_states.size()) {
        //     task_is_active = task_states[progress][id];
        // }
        // if (progress < task_state_history.size()) {
        //     task_is_active = task_state_history[progress];
        // }
        auto recon_progress = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed_time = recon_progress - recon_start;
        if (!restarted && elapsed_time.count() > task_stop_threshold && (allow_restart || id != mpi_root)) {
            task_is_active = 0;
            std::cout << "WARNING: Task " << id << " has stopped." << std::endl;
        }else{
            task_is_active = 1;
        }
        task_state_history.push_back(task_is_active);


        // Do the reconstruction
        if (task_is_active && progress < num_outer_iter && num_rows > 0) {
            std::cout<< "[task-" << id << "]: Outer iteration: " << progress << std::endl;
            // art(data_swap, w_dy, w_dt, w_dx, &center, theta, w_recon, w_ngridx, w_ngridy, num_iter);
            art(local_data, num_rows, dt, dx, &center, theta, local_recon, ngridx, ngridy, num_iter);
        }

        progress++;

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
    
    double total_recon_time, total_comm_time, total_ckpt_time, total_recovery_time;
    MPI_Reduce(&recon_time, &total_recon_time, 1, MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
    MPI_Reduce(&comm_time, &total_comm_time, 1, MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
    MPI_Reduce(&ckpt_time, &total_ckpt_time, 1, MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);
    MPI_Reduce(&recovery_time, &total_recovery_time, 1, MPI_DOUBLE, MPI_SUM, mpi_root, MPI_COMM_WORLD);


    const char * img_name = "recon.h5";
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



    if (id == mpi_root) {
        // Dump the reconsruction configuration and timing to a file
        std::string filePath = "execinfo.json";
        std::ofstream ofile;
        ofile.open(filePath, std::ios::app);

        if (ofile.is_open()) {
            ofile << "{" << std::endl;
            ofile << "\"prob\" : " << failure_prob << "," << std::endl;
            ofile << "\"nslices\" : " << nslices << "," << std::endl;
            ofile << "\"num_iter\" : " << num_outer_iter*num_iter << "," << std::endl;
            ofile << "\"allow_restart\" : " << allow_restart << "," << std::endl;
            ofile << "\"ngridx\" : " << ngridx << "," << std::endl;
            ofile << "\"ngridy\" : " << ngridy << "," << std::endl;
            ofile << "\"theta\" : " << dt << "," << std::endl;
            ofile << "\"total\" : " << total_recon_time/num_tasks << "," << std::endl;
            ofile << "\"ckpt\" : " << total_ckpt_time/num_tasks << "," << std::endl;
            ofile << "\"comm\" : " << total_comm_time/num_tasks << "," << std::endl;
            ofile << "\"recover\" : " << total_recovery_time/num_tasks << "," << std::endl;
            ofile << "}," << std::endl;
        }else{
            std::cerr << "Cannot save the experiment configuration and timing" << std::endl;
        }

        ofile.close();

        std::cout << "ELAPSED TIME: " << total_recon_time/num_tasks << " seconds" << std::endl;
        std::cout << "CHECKPOINTING TIME: " << total_ckpt_time/num_tasks << " seconds" << std::endl;
        std::cout << "COMM TIME: " << total_comm_time/num_tasks << " seconds" << std::endl;
        std::cout << "RECOVERY TIME: " << total_recovery_time/num_tasks << " seconds" << std::endl;
    }


    // free the memory
    delete[] data;
    delete[] data_swap;
    delete[] theta;
    delete[] recon;
    delete[] local_recon;
    delete[] row_indexes;
    delete[] local_data;

    MPI_Finalize();

    return 0;
}
