#!/usr/bin/python

import pyopencl as cl
import numpy
import sys
import cv2
import time
from util import *

# Open the kernel 
kernel = open("kernel_naive.cl").read()

# Convolution kernel 
kernel_dimensions = 5
kernel_sign = 1
kernel_mid = kernel_dimensions / 2
convolution_kernel = gaussian_kernel(kernel_dimensions, kernel_sign)

# Step 1: Create a context.
# This will ask the user to select the device to be used.
context = cl.create_some_context()
# Start timing here after the device is selected
# start_time = time.time()

# Create a queue to the device.
queue = cl.CommandQueue(context)

# Create the program.
program = cl.Program(context, kernel).build()

# Load input image 
img_no_padding = image_to_array(sys.argv[1])
img_no_padding = add_alpha_channel(img_no_padding)
padding = [0, 0 ,0 , 0]
img = cv2.copyMakeBorder(img_no_padding, kernel_mid, kernel_mid, kernel_mid, kernel_mid, cv2.BORDER_CONSTANT, value=padding)
img = pad_image(img, kernel_mid)

(img_h, img_w,depth) = img.shape
flat_img = img.flatten()
flat_kernel = convolution_kernel.flatten()

# Create the result image.
h_output_img = numpy.empty(img_h * img_w * depth).astype(numpy.float32)
h_output_img.fill(0.0)


# Send the data to the guest memory.
d_input_img = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_img)
d_kernel = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=flat_kernel)

# Create the memory on the device to put the result into.
d_output_img = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_output_img.nbytes)

# Initiate the kernel.
conv = program.convolve
conv.set_scalar_arg_dtypes([None, None, None, numpy.uint8, numpy.uint8, numpy.uint32, numpy.uint32, numpy.uint8])

times = numpy.empty(33).astype(numpy.float32)
times.fill(0.0)
for i in range(33):
	start_time = time.time()
	conv(queue, (img_h, img_w), None, d_input_img , d_output_img, d_kernel, kernel_dimensions, kernel_mid, img_w, img_h, depth)
	result = numpy.empty_like(img)
	cl.enqueue_copy(queue, result, d_output_img)
	queue.finish()
	end_time = time.time()
	times[i] = (end_time - start_time) * 1000

queue.finish()
print(times)

sys.exit()
# Wait for the queue to be completely processed.
queue.finish()
# Read the array from the device.
cl.enqueue_copy(queue, h_output_img, d_output_img)
# print(h_output_img)
# end_time = time.time()
print("GPU time = {} ms".format((end_time - start_time) * 1000))
reshaped_img = h_output_img.reshape(img.shape)
# Now remove the padding from image
reshaped_img = reshaped_img[kernel_mid:-kernel_mid]
for i in range(len(reshaped_img)):
    img_no_padding[i] = reshaped_img[i][kernel_mid: -kernel_mid]
save_image(numpy.asarray(img_no_padding).astype(dtype=numpy.uint8), "output_float4.jpg")