/*
    diag_mat_mul

    Multiply diagonal matrix with a full matrix.
    Diagonal matrix is assumed to be stored as a vector.

    Before initialisation the data points is subdivided into
    the the amount of work-groups. These indexes is passed 
    through the n_dat_mat with two values per work-item.


    n_dat_mat - contains indexes of matrices to be used
    mat       - transistion matrix
    diag      - emission matrix

    result    - result of transistion matrix

    TODO:
        Check size of local memory, copy optimal amount of diagonal matrices

    Limitations:
        Matrix size is limited by shared memory size
        for 48kB max size is 78x78.

    Author:
    Gene Stoltz
    ggsgene@gmail.com

*/

__kernel 
void diag_mat_mul(
    __global const int *n_dat_mat,
    __global const float *mat,
    __global const float *diag,
    __global float *result)
{
    // /* Predefined constants
    //     dim             - dimension of the transition matrix
    //     matSize         - dim*dim
    // */

    // Events
    event_t cpy_event;

    // Counters
    int i;
    int j;
    int idx;
    int row;
    int col;

    // Local Variables
    __local float diag_mat_local[dim]; 
    __local float trans_mat_local[matSize]; 
    __local float result_local[matSize];

    // Register Variables
    int id_global;
    int id_local;
    int wrkGroupId;
    int n_dat_local;
    int n_dat_start;
    int idx_diag;
    int idx_dat;
    float prob;
    float trans_mat[dim];
    
    // Get Ids
    id_local  = get_local_id(0);
    id_global = get_global_id(0);
    wrkGroupId = (int)(id_global / dim); 

    // Get number of data points that will be evaluated.
    n_dat_local = n_dat_mat[wrkGroupId + 1] - n_dat_mat[wrkGroupId];
    // Get starting index of data within coherent data structure.
    n_dat_start = n_dat_mat[wrkGroupId];    
    
    // Get transmission matrix into local memory 
    cpy_event = async_work_group_copy(trans_mat_local, mat, matSize, 0);
    wait_group_events(1, &cpy_event); 
    
    // Copy transmission matrix rows related to work-unit to registers
    row = id_local * dim;
    for (i =0; i< dim; ++i)
        trans_mat[i] = trans_mat_local[row + i];

    // Iterate through data points.
    for (j = n_dat_start; j < n_dat_start + n_dat_local; ++j)
    {
        // Index of data result
        idx_dat = (j)*matSize;

        // Index of emission matrix
        idx_diag = (j)*dim;

        // Get emission matrix diagonal
        cpy_event = async_work_group_copy(diag_mat_local,&diag[idx_diag], dim, 0);
        wait_group_events(1, &cpy_event);

        prob = diag_mat_local[id_local];
        // row to multiply in transistion matrix starting index
        row = id_local*dim;
        
        // Iterate through columns in transistion matrix
        for (col = 0; col < dim; ++col)
        {
            result_local[row + col] = prob * trans_mat[col];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        // Copy result to global memory
        cpy_event = async_work_group_copy(&result[idx_dat],result_local, matSize, 0);
        wait_group_events(1, &cpy_event);
    }
}
