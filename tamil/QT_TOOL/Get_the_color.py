import cv2
import numpy as np
import math
import webcolors
from sklearn.cluster import KMeans
from collections import Counter



def get_dominant_color(image, k=4, image_processing_size=None, black=False):
    """
    takes an image as input
    returns the dominant color of the image as a list

    dominant color is found by running k means on the
    pixels & returning the centroid of the largest cluster

    processing time is sped up by working with a smaller image;
    this resizing can be done with the image_processing_size param
    which takes a tuple of image dims as input

    >>> get_dominant_color(my_image, k=4, image_processing_size = (25, 25))
    [56.2423442, 34.0834233, 70.1234123]
    """
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,
                           interpolation=cv2.INTER_AREA)

    # reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    # cluster and assign labels to the pixels
    clt = KMeans(n_clusters=k)
    labels = clt.fit_predict(image)

    # count labels to find most popular
    label_counts = Counter(labels)

    # subset out most popular centroid
    if not black:
        dominant_color = clt.cluster_centers_[label_counts.most_common(2)[1][0]]
    else:
        dominant_color = clt.cluster_centers_[label_counts.most_common(2)[0][0]]

    return list(dominant_color)


def closest_colour(requested_colour):
    min_colours = {}
    for key, name in webcolors.CSS3_HEX_TO_NAMES.items():
        r_c, g_c, b_c = webcolors.hex_to_rgb(key)
        rd = (r_c - requested_colour[0]) ** 2
        gd = (g_c - requested_colour[1]) ** 2
        bd = (b_c - requested_colour[2]) ** 2
        min_colours[(rd + gd + bd)] = name
    return min_colours[min(min_colours.keys())]


def get_colour_name(requested_colour):
    try:
        closest_name = actual_name = webcolors.rgb_to_name(requested_colour)
    except ValueError:
        closest_name = closest_colour(requested_colour)
        actual_name = None
    return actual_name, closest_name


def find_color(img):
    out = get_dominant_color(img, image_processing_size=(100, 100))
    actual_name, closest_name = get_colour_name(tuple(out))
    if actual_name:
        final_name = actual_name
    else:
        final_name = closest_name
    if final_name == 'black':
        out = get_dominant_color(img, image_processing_size=(100, 100), black=True)
        actual_name, closest_name = get_colour_name(tuple(out))
        if actual_name:
            final_name = actual_name
        else:
            final_name = closest_name
    return final_name
