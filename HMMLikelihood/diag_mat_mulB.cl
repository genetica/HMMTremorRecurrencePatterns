/*
    diag_mat_mul

    Multiply diagonal matrix with a full matrix.
    Diagonal matrix is assumed to be stored as a vector.

    Before initialisation the data points is subdivided into
    the the amount of threads. These indexes is passed 
    through the n_dat_mat with two values per work-item.

    Work-items can be scheduled as required, thus no work groups.
    Transistion matrix placed in global cache.

    n_dat_mat - contains indexes of matrices to be used
    mat       - transistion matrix
    diag      - emission matrix

    result    - result of transistion matrix

*/

__kernel 
void diag_mat_mul(
    __global const int *n_dat_mat,
    __global const float *mat,
    __global const float *diag,
    __global float *result)
{
    /* Predefined constants
        dim             - dimension of the transition matrix
    */    

    // Counters
    int i;
    int j;
    int idx;
    int row;
    int col;

    // Defined Variables
    int id_global;
    int n_dat_local;
    int n_dat_start;
    int idx_diag;
    int idx_diag_dim;

    float prob;

    id_global = get_global_id(0);

    // Get number of data points that will be evaluated.
    n_dat_local = n_dat_mat[id_global + 1] - n_dat_mat[id_global];
    // Get starting index of data within coherent data structure.
    n_dat_start = n_dat_mat[id_global];    
    
    // to multiply
    for (j = 0; j < n_dat_local; ++j)
    {
        idx_diag = (n_dat_start + j)*dim;
        idx_diag_dim = idx_diag*dim;
        
        prefetch(&diag[idx_diag],dim);
        for (i = 0; i < dim; ++i)
        {
            prob = diag[idx_diag + i];
            row = i*dim;

            prefetch(&mat[row],dim);
            for (col = 0; col < dim; ++col)
            {
                result[idx_diag_dim + row + col] = (prob * mat[row + col]);
            }
        }
    }


}
