#!/usr/bin/python

import pyopencl as cl
import numpy
import sys
import cv2
import time
from util import *
import os

os.environ['PYOPENCL_NO_CACHE'] = '1'

# Execute
# Gausian kernel
# Convolution kernel
kernel_dim = int(sys.argv[4])
kernel_sign = 1
kernel_mid = kernel_dim / 2
convolution_kernel = gaussian_kernel(kernel_dim, kernel_sign)
kernel_name = sys.argv[2]
local_size = int(sys.argv[3])
img_path = sys.argv[1]
plat = int(sys.argv[5])


# img - fully preprocessed image
# convolution_kernel - opencl kernel

# Load input image
img_no_padding = image_to_array(img_path)

# Add extra channel to enable
img_no_padding = add_alpha_channel(img_no_padding)

# Add padding to the image
padding = [0, 0, 0, 0]
img = cv2.copyMakeBorder(img_no_padding, kernel_mid, kernel_mid,
                         kernel_mid, kernel_mid, cv2.BORDER_CONSTANT, value=padding)
img = pad_image(img, kernel_mid)

# Determine height and width of both original and padded image
(img_h, img_w, depth) = img.shape
(img_original_h, img_original_w, _) = img_no_padding.shape

# Flatten the image and the kernel
flat_img = img.reshape((img_h*img_w*depth))
flat_kernel = convolution_kernel.reshape((kernel_dim * kernel_dim))
# Create the result image.
h_output_img = numpy.empty(img_h * img_w * depth).astype(numpy.uint8)
h_output_img.fill(0)

# opencl kernel
kernel = open(kernel_name + ".cl").read()
# Create context
# Choose a device
platforms = cl.get_platforms()
devices = platforms[plat].get_devices()
context = cl.Context([devices[0]])


# Or choose a device manually
# context = cl.create_some_context()

# Create a queue to the device.
queue = cl.CommandQueue(context)

# Create the program.
program = cl.Program(context, kernel).build()


# Send the data to the guest memory.
d_input_img = cl.Buffer(context, cl.mem_flags.READ_ONLY |
                        cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_img)
d_kernel = cl.Buffer(context, cl.mem_flags.READ_ONLY |
                     cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_kernel)

# Create the memory on the device to put the result into.
d_output_img = cl.Buffer(
    context, cl.mem_flags.WRITE_ONLY, h_output_img.nbytes)

# Initialize the kernel.
conv = program.convolve
conv.set_scalar_arg_dtypes(
    [None, None, None, numpy.int32, numpy.int32, numpy.int32, numpy.int32, numpy.int32])
for i in range(33):
    total_time = 0
    # Execute the kernel
    start_time = time.time()
    conv(queue, (img_original_h, img_original_w), (local_size, local_size), d_input_img,
         d_output_img, d_kernel, kernel_dim, kernel_mid, img_w, img_h, depth, global_offset=[kernel_mid, kernel_mid])
    # Wait for the queue to be completely processed.
    queue.finish()
    # Read the array from the device.
    cl.enqueue_copy(queue, h_output_img, d_output_img)
    end_time = time.time()
    total_time += end_time - start_time
    # stop the time
    print(total_time * 1000)

# Reshape the image array
result_image = h_output_img.reshape(img.shape)

# Now remove the padding from image
result_image = result_image[kernel_mid:-kernel_mid]

for i in range(len(result_image)):
    img_no_padding[i] = result_image[i][kernel_mid:-kernel_mid]

# final img
image = numpy.asarray(img_no_padding).astype(dtype=numpy.uint8)
# verifyImage(image, kernel_mid)

# save_image(image, "output_" + kernel_name + ".jpg")
