__kernel void convolve(
    __global   uchar4* input,
    __global   uchar4* output,
    __constant float* img_kernel,
    const  int kernel_dim,
    const  int kernel_mid,
    const  int width,
    const  int height,
    __local uchar4* localMemory){

    // Get global row and coulmn
    int global_row = get_global_id(0);
    int global_column = get_global_id(1);

    // Get the local dimensions
    int local_row = get_local_id(0);
    int local_column = get_local_id(1);
    // Get the local size
    int local_height = get_local_size(0);
    int local_width = get_local_size(1);
    int index = 0;

    float4 newPixel = (float4)(0.0);
    int image_r_idx, image_c_idx = 0;
    uchar4 pixel = (uchar4)(0);
    // Copy values from global memory to local memory
    localMemory[(local_width*local_row) + (local_column)] = input[width * global_row + global_column];
    // Await all the work item executing this kernel in a workgroup to encounter this function first before continuing
    barrier(CLK_LOCAL_MEM_FENCE);
    // Once data is correctlly copied to local memory,
    // Fetch pixels from local instead of global memory
    // For edge pixels, we make an access to the global memory instead
      

    for(int k_row = 0; k_row < kernel_dim; k_row ++) {
        for(int k_col = 0; k_col < kernel_dim; k_col ++){
            image_r_idx = local_row + (k_row - kernel_mid);
            image_c_idx = local_column + (k_col - kernel_mid);

            if(image_r_idx >= kernel_mid &&
                    image_r_idx <= (local_height - kernel_mid) &&
                    image_c_idx >= kernel_mid &&
                    image_c_idx <= (local_width - kernel_mid)) {
                    index  = (image_r_idx * local_width)  + (image_c_idx);
                    pixel = localMemory[index];
                }else{
                    image_r_idx = global_row + (k_row - kernel_mid);
                    image_c_idx = global_column + (k_col - kernel_mid);
                    index  = (image_r_idx * local_width)  + (image_c_idx);
                    pixel = input[index];
                } 
            
            float4 kernel_value = (float4)img_kernel[k_row * kernel_dim + k_col];
            newPixel += ((float4)((float)pixel.x, (float)pixel.y, (float)pixel.z, (float)pixel.w)) * kernel_value;
        }
    }

    newPixel.w = 0;
    output[global_row * width + global_column] = (uchar4)((uchar)newPixel.x, (uchar)newPixel.y, (uchar)newPixel.z, (uchar)newPixel.w);
    barrier(CLK_LOCAL_MEM_FENCE);
}