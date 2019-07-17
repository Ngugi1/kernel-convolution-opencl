# -*- coding: utf-8 -*-
#!/usr/bin/python
import pyopencl as cl
import numpy
import sys
import cv2
from util import *
# Write down our kernel as a multiline string.
kernel = open("kernel_naive.cl").read()
# Kernel
h_kernel = gaussian_kernel(5, 1)
# h_kernel = blur_kernel()
# Kernel dimensions
h_kernel_size = len(h_kernel)
h_kernel_mid = h_kernel_size / 2
h_kernel_flat = h_kernel.flatten()
# Image without padding
img_no_padding = image_to_array(sys.argv[1])
img_no_padding = add_alpha_channel(img_no_padding)
padding = [255.0, 255.0, 255.0]
# Image with zero padding
img = cv2.copyMakeBorder(image_to_array(
    sys.argv[1]), h_kernel_mid, h_kernel_mid, h_kernel_mid, h_kernel_mid, cv2.BORDER_CONSTANT, value=padding)
img = add_alpha_channel(img)
img = pad_image(img, h_kernel_mid)
# Convert types
(img_h, img_w, bytes_per_pixel) = img.shape

# Step 1: Create a context.
# This will ask the user to select the device to be used.
# Can be automatic, too.
context = cl.create_some_context()
# Create a queue to the device.
queue = cl.CommandQueue(context)
# Create the program.
program = cl.Program(context, kernel).build()
# data to be sent to copied devices
# The output image should be same as original image - no border
h_output_image = numpy.empty_like(img)

# Send the data to the guest memory.
# COPY_HOST_PTR     :: If specified, it indicates that the application wants the
#                      OpenCL implementation to allocate memory for the memory
#                      object and copy the data from memory referenced by host_ptr
# CL_MEM_READ_ONLY  :: This flag specifies that the memory object is a read-only
#                      memory object when used inside a kernel.Writing to a buffer
#                      or image object created with CL_MEM_READ_ONLY inside a kernel is undefined.
# CL_MEM_WRITE_ONLY :: This flags specifies that the memory object will be written but not
#                      read by a kernel.Reading from a buffer or image object created with
#                      CL_MEM_WRITE_ONLY inside a kernel is undefined.
d_input_image = cl.Buffer(context, cl.mem_flags.READ_ONLY |
                          cl.mem_flags.COPY_HOST_PTR, hostbuf=img.flatten())
d_kernel = cl.Buffer(context, cl.mem_flags.READ_ONLY |
                     cl.mem_flags.COPY_HOST_PTR, hostbuf=h_kernel_flat)


# Create the memory on the device to put the result into.
d_output_image = cl.Buffer(
    context, cl.mem_flags.WRITE_ONLY, img.flatten().nbytes)
# Execute the kernel.
# Here you can reference multiple kernels. For didactic purposes I copied the kernel and gave it slightly different name.
# You can reference both kernels here.
convolve = program.convolve
# https://documen.tician.de/pyopencl/runtime_program.html?highlight=set_scalar_arg_dtypes#pyopencl.Kernel.set_scalar_arg_dtypes
convolve.set_scalar_arg_dtypes(
    [None, None, None, numpy.uint8, numpy.uint8, numpy.uint32, numpy.uint32, numpy.uint8])
# https://documen.tician.de/pyopencl/runtime_program.html?highlight=set_scalar_arg_dtypes#pyopencl.Kernel.__call__
convolve(queue, (img_h, img_w), None, d_input_image, d_output_image,
         d_kernel, h_kernel_size, h_kernel_mid, img_w, img_h, bytes_per_pixel)

# Wait for the queue to be completely processed.
# https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clFinish.html
queue.finish()
# Read the array from the device.
# https://documen.tician.de/pyopencl/runtime_memory.html?highlight=enqueue_copy#pyopencl.enqueue_copy
cl.enqueue_copy(queue, h_output_image, d_output_image)

# Output image
reshaped_img = h_output_image.reshape(img.shape)
# Now remove the padding from image
reshaped_img = reshaped_img[h_kernel_mid:-h_kernel_mid]
for i in range(0, (len(reshaped_img) - 1)):
    img_no_padding[i] = reshaped_img[i][h_kernel_mid: -h_kernel_mid]

# Verify the results
print(img_no_padding[0])

save_image(img_no_padding, "output.jpg")

# Verify the results againist the known image
