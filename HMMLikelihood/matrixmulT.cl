/* 

Matrix matrix multiplication
outputting transposed matrices except for computing unit one matrix

    Limitations:
        Matrix size is limited by shared memory size
        for 48kB max size is 78x78.

    Author:
    Gene Stoltz
    ggsgene@gmail.com
*/
// #pragma OPENCL EXTENSION cl_khr_global_int32_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_global_int32_extended_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_local_int32_base_atomics : enable
// #pragma OPENCL EXTENSION cl_khr_local_int32_extended_atomics : enable
__kernel 
void matrixmul(
    __global const int *n_element_mat,
    __global const int *n_mat_mat,
    __global const float *mat,
    __global float *result,
    __global int *result_coeff)
{
    /* Basic matrix multiplication
        Output amount of compute units
        Check matrix overflow and output final coeff
    */

    /* Predefined constants
        dim             - single dimension of matrix
        matSize         - dim x dim
        wrkUnit         - number of work units in work group
        n_elem          - number of elements processed per work unit
        n_mat_mat       - list of matrices to be processed by each work group
        n_element_mat   - list of elements to be processed by each work unit
        minVal          - minimum allowed value before scaling.
    */
    // maxVal should be changed to be an input.
    const float maxVal = 1e6;

    const float MIN_PRECISION = 1e-64;
    // Defined Variables
    __local float input[matSize]; 
    __local float output[matSize];
    __local int coeff[1];
    __local int finalcoeff[1];
    __local float currentmax[wrkUnit];
    
    float output_temp[n_elem];

    int id_global = get_global_id(0);
    int id_local  = get_local_id(0);
    
    int wrkGroupId = (int)(id_global / wrkUnit); 

    // Counters
    int i;
    int j;
    int m;
    int n;
    int row;
    int col;
    int idx;

    int n_elem_local= n_element_mat[id_local + 1] - n_element_mat[id_local];
    int n_elem_start = n_element_mat[id_local];

    int n_mat_local = n_mat_mat[wrkGroupId + 1] - n_mat_mat[wrkGroupId];
    int n_mat_start = n_mat_mat[wrkGroupId];

    if (id_local == 0)
    {
        coeff[0] = 0;
        finalcoeff[0] = 0;
        //printf("S%d",id_global);
    }

    // Initial matrix copy
    for (i = 0 ; i < n_elem_local; ++i)
    {
        idx = n_elem_start + i;
        output[idx] = mat[n_mat_start*matSize + idx];
    }

    barrier(CLK_LOCAL_MEM_FENCE);

    // to multiply
    for (j = 1; j < n_mat_local; ++j)
    {
        prefetch(&mat[(n_mat_start + j)*matSize + n_elem_start],n_elem_local);
        // Copy new matrix
        if (coeff[0] != 0)
        {
            for (i = 0 ; i < n_elem_local; ++i)
            {
                idx = n_elem_start + i;
                input[idx] = mat[(n_mat_start + j)*matSize + idx] * exp2((float)coeff[0]);
                //printf("%d\n", coeff[0]);
            }
        }
        else
        {
            for (i = 0 ; i < n_elem_local; ++i)
            {
                idx = n_elem_start + i;
                input[idx] = mat[(n_mat_start + j)*matSize + idx];
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        
        for (i = 0 ; i < n_elem_local; ++i)
        {
            idx = n_elem_start + i;
            col = idx % dim;
            row = (int)(idx / dim);
            
            output_temp[i] = 0;
            for (m = 0; m < dim; ++m)
            {
                output_temp[i] += output[row*dim + m]*input[m*dim + col];
            }
        }

        if (id_local == 0)
        {
            if (coeff[0] != 0){
                finalcoeff[0] += (coeff[0]);
                coeff[0] = 0;
            }
        }

        barrier(CLK_LOCAL_MEM_FENCE);
        currentmax[id_local] = MIN_PRECISION;
        for (i = 0 ; i < n_elem_local; ++i)
        {
            idx = n_elem_start + i;
            
            if (output_temp[i] > currentmax[id_local])
            {
                //atomic_or(&coeff[0],1);
                currentmax[id_local] = output_temp[i];
            }
            if (output_temp[i] < MIN_PRECISION)
                output[idx] = 0;
            else
                output[idx] = output_temp[i];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
        if (id_local == 0) //&& (coeff[0] != 0))
        {
            float maxCurrent = MIN_PRECISION;//(float)minVal;
            for (i =0; i < wrkUnit; ++i)
            {
                if (((float)fabs((float) currentmax[i]) > maxCurrent))
                {
                    maxCurrent = (float)fabs((float)currentmax[i]);
                }
            }

            if (maxCurrent < minVal)
            {
                float temp1 = log2((float)(minVal/2) / maxCurrent + 1);
                coeff[0] = (int)temp1;
                // printf("M %e %d \n",t,coeff[0]);
                // if (t < 1e-10)
                // {
                //     for (m = 0; m <= matSize; ++m)
                //     printf("%f ",result[wrkGroupId*matSize + m]);
                // }
            }
            else
            // {
            //     coeff[0] = 0;
            // }

            if (maxCurrent > maxVal)
            {
                float temp2 = log2((float)maxCurrent / (maxVal/2) + 1);
                coeff[0] = (int)temp2 * -1;
            }
            else
            {
                coeff[0] = 0;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    
    // Output transposed matrix except for global_id 0
    // if (id_global == 0)
    // {
        // Non transposed matrix
        for (i = 0 ; i < n_elem_local; ++i)
        {
            idx = n_elem_start + i;
            result[wrkGroupId*matSize + idx] = output[idx];
        }
    // }
    // else
    // {
    //     // transposed matrix
    //     for (i = 0 ; i < n_elem_local; ++i)
    //     {
    //         idx = n_elem_start + i;
    //         result[wrkGroupId*matSize + idx] = output[idx];
    //     }
    // }
    if (id_local == 0)
    {
        result_coeff[wrkGroupId] = finalcoeff[0];
    }
}
