#!/usr/bin/python

import pyopencl as cl
import numpy

# Write down our kernel as a multiline string.
kernel = """
__kernel void vadd(
    __global float* a,
    __global float* b,
    __global float* c,
    const unsigned int count)
{
    unsigned int i = get_global_id(0);
    if (i < count)
        c[i] = a[i] + b[i];
}

__kernel void vaddd(
    __global float* a,
    __global float* b,
    __global float* c,
    const unsigned int count)
{
    unsigned int i = get_global_id(0);
    if (i < count)
        c[i] = a[i] + b[i];
}
"""

# The size of the vectors to be added together.
vector_size = 1024

# Step 1: Create a context.
# This will ask the user to select the device to be used.
# Can be automatic, too.
context = cl.create_some_context()

# Create a queue to the device.
queue = cl.CommandQueue(context)

# Create the program.
program = cl.Program(context, kernel).build()

# Create two vectors to be added.
h_a = numpy.random.rand(vector_size).astype(numpy.float32)
h_b = numpy.random.rand(vector_size).astype(numpy.float32)

# Create the result vector.
h_c = numpy.empty(vector_size).astype(numpy.float32)

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
d_a = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_a)
d_b = cl.Buffer(context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=h_b)

# Create the memory on the device to put the result into.
d_c = cl.Buffer(context, cl.mem_flags.WRITE_ONLY, h_c.nbytes)

# Execute the kernel.
# Here you can reference multiple kernels. For didactic purposes I copied the kernel and gave it slightly different name.
# You can reference both kernels here.
vadd = program.vaddd

# https://documen.tician.de/pyopencl/runtime_program.html?highlight=set_scalar_arg_dtypes#pyopencl.Kernel.set_scalar_arg_dtypes
vadd.set_scalar_arg_dtypes([None, None, None, numpy.uint32])

# https://documen.tician.de/pyopencl/runtime_program.html?highlight=set_scalar_arg_dtypes#pyopencl.Kernel.__call__
vadd(queue, h_a.shape, None, d_a, d_b, d_c, vector_size)

# Wait for the queue to be completely processed.
# https://www.khronos.org/registry/OpenCL/sdk/1.0/docs/man/xhtml/clFinish.html
queue.finish()

# Read the array from the device.
# https://documen.tician.de/pyopencl/runtime_memory.html?highlight=enqueue_copy#pyopencl.enqueue_copy
cl.enqueue_copy(queue, h_c, d_c)

# Verify the solution.
correct = 0
tolerance = 0.001

for i in range(vector_size):
    # Expected result
    expected = h_a[i] + h_b[i]
    actual = h_c[i]
    # Compute the relative error
    relative_error = numpy.absolute((actual - expected) / expected)

    # Print the index if it's wrong.
    if relative_error < tolerance:
        correct += 1
    else:
        print i, " is wrong"
print(h_c)