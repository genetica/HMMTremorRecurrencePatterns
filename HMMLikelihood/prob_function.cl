/*

kernel_function

This is the kernel function executed by diag_normalc.cl

The text are copied into diag_normalc.cl and then 
sent to be compiled.

float* param
an array of the parameters defined to support the kernel function.

float* input
an array of the input data defined

return
the likelihood of the data with regards to the input data point

*/ 

float kernel_function(float* param, float* input)  
{
    float p    = param[5];
    int z      = (int)input[2];

    if (z == 0) {
        return 1 - p;
    }
    else {
        float f;
        float mu_x    = param[0];
        float mu_y    = param[1];
        float sigma_x = param[2];
        float sigma_y = param[3];
        float rho     = param[4];

        float x = input[0];
        float y = input[1];

        float b1 = pown(x - mu_x, 2) / pown(sigma_x, 2);
        float b2 = pown(y - mu_y, 2) / pown(sigma_y, 2);
        float b3 = 2 * rho * (x - mu_x)*(y - mu_y) / (sigma_y *sigma_x);
        float b4 = -1 / (2 * (1 - pown(rho,2)));
        float b5 = 1 / (2 * M_PI*sigma_x * sigma_y * sqrt(1 - pown(rho,2)) );

        f = b5 * exp(b4 * (b1 + b2 - b3));    
        return p*f;
    }
}

