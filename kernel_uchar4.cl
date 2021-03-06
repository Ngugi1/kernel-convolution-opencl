__kernel void convolve(
    __global   uchar4* input,
    __global   uchar4* output,
    __constant float* img_kernel,
    int kernel_dim,
    int kernel_mid,
    int width,
    int height,
    int depth){ // Depth is not used here, retained for API convinience
    // Ignore the pixels used for padding 
    int row = get_global_id(0);
    int column = get_global_id(1);
    float4 newPixel = (float4)(0);
    for(int k_row = 0; k_row < kernel_dim; k_row ++) {
        for(int k_col = 0; k_col < kernel_dim; k_col ++){  
            // X and Y of pixel on which kernel maps.
            int image_r_idx = row + (k_row - kernel_mid);
            int image_c_idx = column + (k_col - kernel_mid);
            int index  = (image_r_idx * width)  + (image_c_idx);
            float4 kernel_value = (float4)img_kernel[k_row * kernel_dim + k_col];
            uchar4 pixel = input[index];
            newPixel += ((float4)((float)pixel.x, (float)pixel.y, (float)pixel.z, (float)pixel.w)) * kernel_value;
        }
    }
    // Calculate output index
    newPixel.w = 0;
    output[row * width + column] = (uchar4)((uchar)newPixel.x, (uchar)newPixel.y, (uchar)newPixel.z, (uchar)newPixel.w);
}