__kernel void convolve(
    __global   uchar4* input,
    __global   uchar4* output,
    __constant float* img_kernel,
    const  int kernel_dim,
    const  int kernel_mid,
    const  int width,
    const  int height,
    const  int bytes_per_pixel,
    __local uchar4* localMemory){

    int global_row = get_global_id(0);
    int global_column = get_global_id(1);
    // Get the local dimensions
    int local_row = get_local_id(0);
    int local_column = get_local_id(1);
    // Get the local size
    int local_height = get_local_size(0);
    int local_width = get_local_size(1);
    // Copy values from global memory to local memory
    localMemory[(local_width*local_row) + (local_column)] = input[width * global_row + global_column];
    // Await all the work item executing this kernel in a workgroup to encounter this function first before continuing
    barrier(CLK_LOCAL_MEM_FENCE);
    // Once data is correctlly copied to local memory,
    // Fetch pixels from local instead of global memory
    // For edge pixels, we make an access to the global memory instead
    // Ignore edge pixels at the top level
    if(global_column >= kernel_mid && global_column < (width - kernel_mid)
    && global_row >= kernel_dim && global_row < (height - kernel_mid)){
        float4 newPixel = (float4)(0);
        for(int k_row = 0; k_row < kernel_dim; k_row ++) {
            for(int k_col = 0; k_col < kernel_dim; k_col ++){  
                // X and Y of pixel on which kernel maps.
                int image_r_idx = 0;
                int image_c_idx = 0;
                uchar4 pixel = (uchar4)(0);
                float4 kernel_value = (float4)img_kernel[k_row * kernel_dim + k_col];
                if(local_row >= kernel_mid &&
                    local_row < local_height &&
                    local_column >= kernel_mid &&
                    local_column < local_width) {
                    image_r_idx = local_row + (k_row - kernel_mid);
                    image_c_idx = local_column + (k_col - kernel_mid);
                    int index  = (image_r_idx * width)  + (image_c_idx);
                    pixel = localMemory[index];
                }else{
                    image_r_idx = global_row + (k_row - kernel_mid);
                    image_c_idx = global_column + (k_col - kernel_mid);
                    int index  = (image_r_idx * width)  + (image_c_idx);
                    pixel = input[index];
                }
                newPixel += ((float4)((float)pixel.x, (float)pixel.y, (float)pixel.z, (float)pixel.w)) * kernel_value;
            }
        }
    output[global_row * width + global_column] = (uchar4)((uchar)newPixel.x, (uchar)newPixel.y, (uchar)newPixel.z, (uchar)newPixel.w);
    }
    barrier(CLK_LOCAL_MEM_FENCE);
}