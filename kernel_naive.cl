__kernel void convolve(
    __global unsigned char* input,
    __global unsigned char* output,
    __global float* img_kernel,
    const  int kernel_dim,
    const  int kernel_mid,
    const  int width,
    const  int height,
    const  int bytes_per_pixel){
    
    int row = get_global_id(0);
    int column = get_global_id(1);
    float sumRed = 0.0;
    float sumGreen = 0.0;
    float sumBlue = 0.0;
    // Ignore pixels in the padding
    if (row >= kernel_mid && row < height - kernel_mid && column >= kernel_mid && column < width - kernel_mid){
         for(int k_row = 0; k_row < kernel_dim; k_row ++) {
             for(int k_col = 0; k_col < kernel_dim; k_col ++){  
                // X and Y of pixel on which kernel maps.
                int image_r_idx = row + (k_row - kernel_mid);
                int image_c_idx = column + (k_col - kernel_mid);
                float kernel_value = img_kernel[k_row * kernel_dim + k_col];
                int index_on_flat_img  = (image_r_idx * width * bytes_per_pixel)  + (image_c_idx * bytes_per_pixel);
                sumRed = sumRed + (kernel_value * input[index_on_flat_img]);
                sumGreen = sumGreen + (kernel_value * input[index_on_flat_img + 1]);
                sumBlue = sumBlue + (kernel_value * input[index_on_flat_img + 2]);
            }
        }
        // Calculate output index
        int output_index = (row * width * bytes_per_pixel) + (column * bytes_per_pixel);
        output[output_index] = sumRed;
        output[output_index + 1] = sumGreen;
        output[output_index + 2] = sumBlue;
        output[output_index + 3] = 0;
    }
}