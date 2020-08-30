import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import math
from shapely.geometry import Polygon
from skimage.io import imread


def get_length(img, cm_to_pixel):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    contour_bound = c.tolist()
    contour_bound = [i[0] for i in contour_bound]
    top_most_point = min(c.tolist(), key=lambda x: x[0][1])[0]
    bottom_most_point = max(c.tolist(), key=lambda x: x[0][1])[0]
    p1 = top_most_point
    p2 = bottom_most_point

    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))

    # veggi_len = arch_len / 2
    veggi_len = round(distance * cm_to_pixel, 2)
    return veggi_len


def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [px, py]


def get_width(img, cm_to_pixel):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    contour_bound = c.tolist()
    contour_bound = [i[0] for i in contour_bound]
    top_most_point = min(c.tolist(), key=lambda x: x[0][1])[0]
    bottom_most_point = max(c.tolist(), key=lambda x: x[0][1])[0]
    mid_contour = midpoint(top_most_point, bottom_most_point)

    intersection_points = []
    for i in range(len(contour_bound) - 1):
        x1 = contour_bound[i]
        x2 = contour_bound[i + 1]
        x3 = [0, mid_contour[1]]
        x4 = [mid_contour[0] * 5, mid_contour[0]]
        line = LineString([x1, x2])
        other = LineString([x3, x4])
        if line.intersects(other):
            point = list(line.intersection(other).coords)[0]
            intersection_points.append(point)

    p1 = intersection_points[0]
    p2 = intersection_points[1]

    distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
    distance = round(distance * cm_to_pixel, 2)

    return distance


def lerp(v0, v1, i):
    return v0 + i * (v1 - v0)


def getEquidistantPoints(p1, p2, n):
    return [[lerp(p1[0], p2[0], 1. / n * i), lerp(p1[1], p2[1], 1. / n * i)] for i in range(n + 1)]


def get_max_width(img, cm_to_pixel):
    """

    This is function is to identify the maximum width of the contour. It finds the topmost and bottom-most points present
    in the contour and it will interpolate 100 points in between. Each point will draw a horizontal line that
    will intersect with the contour. Those two intersection points will return the width of the contour at that
    particular point. It will calculate the width for all 100 points and return the maximum width on the list.

    :param img: this needs to be binary image
    :param cm_to_pixel: This value represents the sampling distance of a single pixel in cm unit
    :return: Retrun the Maximum width of the maximum contour preset in the binary image
    """
    contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    contour_bound = c.tolist()
    contour_bound = [i[0] for i in contour_bound]
    top_most_point = min(c.tolist(), key=lambda x: x[0][1])[0]
    bottom_most_point = max(c.tolist(), key=lambda x: x[0][1])[0]
    interpolated_points = getEquidistantPoints(top_most_point, bottom_most_point, 100)

    widths = []
    for wid in interpolated_points:
        mid_contour = wid

        intersection_points = []
        for i in range(len(contour_bound) - 1):
            x1 = contour_bound[i]
            x2 = contour_bound[i + 1]
            x3 = [0, mid_contour[1]]
            x4 = [mid_contour[0] * 5, mid_contour[0]]
            line = LineString([x1, x2])
            other = LineString([x3, x4])
            if line.intersects(other):
                point = list(line.intersection(other).coords)[0]
                intersection_points.append(point)

        p1 = intersection_points[0]
        p2 = intersection_points[1]

        distance = math.sqrt(((p1[0] - p2[0]) ** 2) + ((p1[1] - p2[1]) ** 2))
        distance = round(distance * cm_to_pixel, 2)
        widths.append(distance)

    max_width = max(widths)
    return max_width


def animate(img, crop_img, length, width, color, coord):
    gray = cv2.cvtColor(crop_img, cv2.COLOR_RGB2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    for i in c:
        i[0][0] = i[0][0] + coord[0]
        i[0][1] = i[0][1] + coord[1]
    cv2.drawContours(img, c, -1, (0, 0, 255), 3)
    contour_bound = c.tolist()
    contour_bound = [i[0] for i in contour_bound]
    P = Polygon(contour_bound)
    centre_point = list(P.centroid.coords)[0]
    # h, w = 50 / 2, 50 / 2
    # minx, miny = centre_point[0] - w, centre_point[1] - h
    # maxx, maxy = centre_point[0] + w, centre_point[1] + h
    #
    # fontScale = (img.shape[0] * img.shape[1]) / (3000 * 3000)
    # cv2.rectangle(img, (minx, miny), (maxx, maxy), (0, 0, 0), 2)
    # # Number to object
    # cv2.putText(img, str(i + 1), (int(roi[0] + roi[2] / 2), int(roi[1] + roi[3] / 2)), cv2.FONT_HERSHEY_DUPLEX,
    #             2 * fontScale, text_color, 1)
    # # Write the Color and length
    # cv2.putText(img, str(length * 10) + " cm", (roi[0] + 1, roi[1] + roi[3] + int(50 * fontScale)),
    #             cv2.FONT_HERSHEY_DUPLEX, fontScale, text_color, 1)
    # cv2.putText(img, str(color), (roi[0], roi[1] + roi[3] + int(100 * fontScale)), cv2.FONT_HERSHEY_DUPLEX, fontScale,
    #             text_color, 1)
    # cv2.putText(img, "Hue : " + str(color_hsv[0][0][0]), (roi[0], roi[1] + roi[3] + int(150 * fontScale)),
    #             cv2.FONT_HERSHEY_DUPLEX, fontScale, text_color, 1)