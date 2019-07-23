__kernel void convolve(
    __global  const float4* input,
    __global  float4* output,
    __global  const float* img_kernel,
    const  int kernel_dim,
    const  int kernel_mid,
    const  int width,
    const  int height,
    const  int bytes_per_pixel){
    
    // Ignore the pixels used for padding 
    int row = get_global_id(0) + kernel_mid;
    int column = get_global_id(1) + kernel_mid;
    float4 newPixel = (float4)(0.0);
    
    for(int k_row = 0; k_row < kernel_dim; k_row ++) {
        for(int k_col = 0; k_col < kernel_dim; k_col ++){  
            // X and Y of pixel on which kernel maps.
            int image_r_idx = row + (k_row - kernel_mid);
            int image_c_idx = column + (k_col - kernel_mid);
            int index  = (image_r_idx * width)  + (image_c_idx);
            float kernel_value = img_kernel[k_row * kernel_dim + k_col];
            float4 pixel = input[index];
            newPixel += pixel * kernel_value;
        }
    }
    // Calculate output index
    output[row * width + column] = newPixel;
}