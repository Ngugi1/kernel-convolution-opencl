# -*- coding: utf-8 -*-
# This file contains utility methods to manipulate images and verify output
import numpy
import sys
import math
import cv2
from PIL import Image
numpy.set_printoptions(threshold=numpy.nan)  # print all data

#################
# Image helpers #
#################

def verifyImage(img, kernel_mid):
    (h, w, d) = img.shape
    correct = 0
    wrong = 0
    tolerance = 5
    seq_image = add_alpha_channel(image_to_array("output_seq.jpg"))
    for i in range(kernel_mid, (h - kernel_mid)):
        for j in range(kernel_mid, (w - kernel_mid)):
            seq_pixel = seq_image[i][j]
            other_pixel = img[i][j]
            # Use the red byte to check for consistency 
            if(seq_pixel[0] == 0):
                relative_error = 0
                correct = correct + 1
            else:
                relative_error = numpy.absolute((seq_pixel[0] - other_pixel[0]) / seq_pixel[0])
                # Print the index if it's wrong.
                if relative_error < tolerance:
                    correct += 1
                else:
                    wrong +=1
    print("correct == {}, Wrong = {}".format(correct, wrong))







def pad_image(arr, kernel_mid):
    """
    Pad the original image with pixels on the border
    Args:
        arr       : the numpy array
        kernel_mid: the kernel mid
    Returns:
        the extended numpy array
    """
    (height, width, _bytes_per_pixel) = arr.shape
    for i in range(kernel_mid):
        "Dublicate the top row"
        arr[i] = arr[kernel_mid]
    for i in range(height - kernel_mid, height):
        "Duplicate the last row"
        arr[i] = arr[height - kernel_mid - 1]
    for i in range(height):
        "Duplicate the edge columns"
        for j in range(kernel_mid):
            "Duplicate left"
            arr[i][j] = arr[i][kernel_mid]
        for k in range(width - kernel_mid, width):
            "Duplicate right"
            arr[i][k] = arr[i][width - kernel_mid - 1]
    return arr.astype(dtype=numpy.uint8)


def add_alpha_channel(arr):
    # Split the image to 4 channels
    b_channel, g_channel, r_channel = cv2.split(arr)
    # creating a dummy alpha channel image.
    alpha_channel = numpy.zeros(b_channel.shape, dtype=b_channel.dtype)
    img_BGRA = cv2.merge((b_channel, g_channel, r_channel, alpha_channel))
    return img_BGRA


def save_image(arr, filename):
    """Takes a numpy array, in proper shape, and writes is an image
    Args:
        arr     : numpy array in shape (x, y, rgb)
        filename: name of the image to write
    """
    img = Image.fromarray(arr, "RGBA")
    img.save(filename)


def image_to_array(file):
    """Read an image into a numpy 3d array
    Args:
        file: filepath to image
    Returns:
        A 3d numpy array of type float32.
    """
    img = Image.open(file)
    # Convert the image to a numpy matrix.
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
    arr = (numpy.array(
        [[1.0, 2.0, 1.0], [2.0, 4.0, 2.0], [1.0, 2.0, 1.0]])) * 1 / 16
    return normalize_kernel(arr, 3)
