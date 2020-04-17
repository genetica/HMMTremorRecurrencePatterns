/*
 
 Require to add a function kernel_function

*/

__kernel 
void prob_test(
    __global const float *paramin,
    __global const float *datain,
    __global float *result)
{
    float param[n_kernel_param];
    float data[n_data_dim];

    //[n_kernel_param];
    //[n_data_dim];
    int i, j;
    for (i=0; i< n_kernel_param; ++i)
        param[i] = paramin[i];
    for (i=0; i< n_data_dim; ++i)
        data[i] = datain[i];

    // Defined Variables
    int id_global = get_global_id(0);

    result[0] = kernel_function(param,data);
}
