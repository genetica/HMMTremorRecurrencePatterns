float kernel_function(float* param, float* data)  
{
    float f;
    float mu_x    = param[0];
    float sigma_x = param[2];

    float x = data[0];

    float b1 = pown( (x - mu_x) / sigma_x, 2  );
    float b2 = 1 / ( sigma_x * sqrt(2 * M_PI) );

    f = b2 * exp(-0.5 * b1);
    return f;
}

