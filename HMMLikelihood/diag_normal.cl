/*
 
 Require to add a function kernel_function

*/

__kernel 
void diag_normal(
    __global const float *kern,
    __global const float *datain,
    __global float *result)
{
    /* Predefined constants
        matSize
        dim
        wrkUnit
        n_kernel_param
        n_data_dim
    */
    // Defined Variables
    int id_global = get_global_id(0);
    float param[n_kernel_param];
    float data[n_data_dim];

    // Counters
    int i;
    int j;

    // Get data point data
    for (i = 0; i < n_data_dim; ++i)
        data[i] = datain[n_data_dim*id_global + i];

    // Start position for data point 
    int  n_dat_start = id_global * dim;
    for (j = 0; j < dim; ++j)
    {
        // Get current kernel parameters
        for (i = 0; i < n_kernel_param; ++i)
            param[i] = kern[n_kernel_param*j + i];

        result[n_dat_start + j] = kernel_function(param,data);
    }
}
