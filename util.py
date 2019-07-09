# -*- coding: utf-8 -*-
# This file contains utility methods to manipulate images and verify output
import numpy
import sys
import math

from PIL import Image
numpy.set_printoptions(threshold=numpy.nan) # print all data

#################
# Image helpers #
#################

def save_image(arr, filename):
    """Takes a numpy array, in proper shape, and writes is an image
    Args:
        arr     : numpy array in shape (x, y, rgb)
        filename: name of the image to write
    """
    img = Image.fromarray(arr, "RGB")
    img.save(filename)


def image_to_array(file):
    """Read an image into a numpy 3d array
    Args:
        file: filepath to image
    Returns:
        A 3d numpy array of type uint8.
    """
    img = Image.open(file)
    # Convert the image to a numpy matrix of unsigned 8 bit integers.
    img_arr = numpy.asarray(img).astype(numpy.uint8)
    return img_arr


def flat_array_to_image(arr, dims, filename):
    """ Write a 2d array to an image
    Args:
        arr     : The numpy array that contains the data
        dims    : a triple containing the rows, columns and depth of the image respectively
        filename: filepath to write the imae to
    """
    reshaped = arr.reshape(dims)
    save_image(reshaped, filename)

##################
# Kernel Helpers #
##################

def normalize_kernel(kernel, dim):
    """Normalizes a kernel
    Args:
        kernel: a two-d kernel
    """
    for x in range(0, dim):
        for y in range(dim):
            kernel[x][y] = kernel[x][y] / numpy.sum(kernel)
    return kernel


def gaussian_kernel(dim, sigma):
    """
    The Guassian blur function is as follows:

                           x² + y²
    G(x,y) =    1        - -------
            --------- * e    2σ²
              2πσ²
    Finally the kernel is normalized to avoid too dark or too light areas.
    """
    rows = dim
    cols = dim
    arr = numpy.empty([rows, cols]).astype(numpy.float32)
    center = dim / 2
    total = 0.0
    for x in range(0, rows):
        for y in range(0, cols):
            x_ = x - center
            y_ = y - center
            arr[x][y] = (1 / (2.0 * math.pi * math.pow(sigma, 2))) * math.pow(math.e, -1.0 * (
                (math.pow(x_, 2) + math.pow(y_, 2)) / (2.0 * math.pow(sigma, 2))))
            total = total + arr[x][y]

    return normalize_kernel(arr, dim)


def identity_kernel(dim):
    """ The identity kernel
    0 0 0
    0 1 0
    0 0 0
    """
    arr = numpy.empty([dim, dim]).astype(numpy.float32)
    arr.fill(0.0)
    arr[dim / 2][dim / 2] = 1.0
    return normalize(numpy.array([[-1.0, 0.0, 1.0], [-2.0, 0, 2.0], [-1.0, 0.0, -1.0]]), 3)


def blur_kernel():
    """ Blurring kernel
    1 2 1
    2 4 2 * 1/16
    1 2 1
    """
    arr = (numpy.array([[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])) * 1 / 16
    return normalize_kernel(arr, 3)