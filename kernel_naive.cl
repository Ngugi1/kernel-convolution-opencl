kernel void convolve(
    global unsigned char* input,
    global unsigned char* output,
    global float* img_kernel,
      int kernel_dim,
      int kernel_mid,
      int width,
      int height,
      int depth){
    
    // Ignore pixels used for padding
    int row = get_global_id(0);
    int column = get_global_id(1);

    float sumRed = 0.0;
    float sumGreen = 0.0;
    float sumBlue = 0.0;
    // Ignore pixels in the padding
      for(int k_row = 0; k_row < kernel_dim; k_row ++) {
             for(int k_col = 0; k_col < kernel_dim; k_col ++){  
                // X and Y of pixel on which kernel maps.
                int image_r_idx = row + (k_row - kernel_mid);
                int image_c_idx = column + (k_col - kernel_mid);
                float kernel_value = img_kernel[k_row * kernel_dim + k_col];
                int index_on_flat_img  = (image_r_idx * width * depth)  + (image_c_idx * depth);
                sumRed += (kernel_value * input[index_on_flat_img]);
                sumGreen += (kernel_value * input[index_on_flat_img + 1]);
                sumBlue += (kernel_value * input[index_on_flat_img + 2]);
            }
        }
        // Calculate output index
        unsigned int output_index = (row * width * depth) + (column * depth);
        output[output_index] = sumRed;
        output[output_index + 1] = sumGreen;
        output[output_index + 2] = sumBlue;
        output[output_index + 3] = 0;
}