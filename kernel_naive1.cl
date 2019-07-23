__kernel void convolve(
    __global const float* input,
    __global float* output,
    __global const float* img_kernel,
    const  int kernel_dim,
    const  int kernel_mid,
    const  int width,
    const  int height,
    const  int bytes_per_pixel){

    int row = get_global_id(0);
    int col = get_global_id(1);
    
    int index = (row * width * bytes_per_pixel) + (col * bytes_per_pixel);

    output[index] = input[index];
    output[index + 1] = input[index + 1];
    output[index + 2] = input[index + 2];
    output[index + 3] = input[index + 3];
}