from numpy import float32
from configure import RESOURCE_PATH

try:
    import os
    import json
    import glob
    import argparse

    import numpy as np
    from scipy import signal as sg
    from scipy.ndimage.filters import maximum_filter

    from PIL import Image

    import matplotlib.pyplot as plt
except ImportError:
    print("Need to fix the installation")
    raise


class Suspicious_points_tfl:

    def __init__(self, image=""):
        self.image = image

    def set_image(self, image):
        self.image = image

    def get_kernel_circle(self, kernel_size):
        """Get the desired size, open the relevant image and create a kernel from it"""
        circle_img = Image.open(f'{RESOURCE_PATH}/{kernel_size}Kernel.png')
        kernel_circle = np.asarray(circle_img)[:, :, 0]  # take one layer
        kernel_circle = kernel_circle.astype(float32)
        kernel_circle -= 100  # The black dots will be negative and the white would remain positive.
        sum_circle = np.sum(kernel_circle)
        # Calculate the size and normal the kernel to sum of one.
        area = circle_img.width * circle_img.height
        kernel_circle -= (sum_circle / area)
        max_kernel_circle = np.max(kernel_circle)
        kernel_circle /= max_kernel_circle
        return kernel_circle

    def get_dots(self, image, threshold):
        """Get an image and threshold and return the filtered dots."""
        c_image_converted = image.astype(np.float32)[10:-5, 10:-10]
        filtered = maximum_filter(c_image_converted, 50)
        lights_indices = np.argwhere((threshold < c_image_converted) & (filtered == c_image_converted))
        return lights_indices + 10

    def remove_duplicate_points(self, points):
        points = points.tolist()
        by_x = sorted(points, key=lambda x: (x[0], x[1]))
        remove_x = []
        i = 0
        while i < len(by_x):
            remove_x.append(by_x[i])
            i += 1
            for j in range(i, len(by_x)):
                if by_x[j][0] - remove_x[-1][0] <= 10 and by_x[j][1] - remove_x[-1][1] <= 10 :
                    i+=1
                else:
                    break

        by_y = sorted(remove_x, key=lambda x: (x[1], x[0]))
        remove_y = []
        i = 0
        while i < len(by_y):
            remove_y.append(by_y[i])
            i += 1
            for j in range(i, len(by_y)):
                if by_y[j][0] - remove_y[-1][0] <= 10 and by_y[j][1] - remove_y[-1][1] <= 10:
                    i += 1
                else:
                    break

        return np.array(remove_y)

    def find_tfl_by_layer(self, c_image, layer, t_small, t_big):
        """Get an image and layer and the threshold and return the attention dots"""
        c_image_bw = Image.fromarray(c_image)
        c_image_bw_array = np.asarray(c_image_bw)[:, :, layer]
        c_image_bw_array = c_image_bw_array.astype(float32)

        # find_small_tfl
        kernel_circle = self.get_kernel_circle("small")
        res = sg.convolve(c_image_bw_array, kernel_circle, mode='same', method='auto')
        threshold = np.max(res[10:-5, 10:-10]) - t_small  # Filter the frame dots.
        dots_small = self.get_dots(res, threshold)

        # find_big_tfl
        kernel_circle = self.get_kernel_circle("big")
        res = sg.convolve(c_image_bw_array, kernel_circle, mode='same', method='auto')
        threshold = np.max(res[10:-5, 10:-10]) - t_big  # Filter the frame dots.
        dots_big = self.get_dots(res, threshold)
        dots = np.vstack([dots_small, dots_big])
       # dots = self.remove_duplicate_points(dots)
        return dots

    def find_tfl_lights(self, c_image: np.ndarray, **kwargs):
        """
        Detect candidates for TFL lights. Use c_image, kwargs and you imagination to implement
        :param c_image: The image itself as np.uint8, shape of (H, W, 3)
        :param kwargs: Whatever config you want to pass in here
        :return: 4-tuple of x_red, y_red, x_green, y_green
        """
        dots_red_layer = self.find_tfl_by_layer(c_image, 0, 1000, 5000)
        dots_green_layer = self.find_tfl_by_layer(c_image, 1, 1000, 5000)
        # Filter the attention dots to Green and Red.
        filtered_red_dots, filtered_green_dots = np.array([[0, 0]]), np.array([[0, 0]])
        for dot in dots_red_layer:
            if c_image[dot[0], dot[1]][0] >= c_image[dot[0], dot[1]][1]:
                filtered_red_dots = np.append(filtered_red_dots, [[dot[0], dot[1]]], axis=0)
        for dot in dots_green_layer:
            if c_image[dot[0], dot[1]][1] > c_image[dot[0], dot[1]][0]:
                filtered_green_dots = np.append(filtered_green_dots, [[dot[0], dot[1]]], axis=0)

        return filtered_red_dots[1:, :], filtered_green_dots[1:, :]

    def run(self):
        image = np.array(Image.open(self.image))

        return self.find_tfl_lights(image, some_threshold=42)
