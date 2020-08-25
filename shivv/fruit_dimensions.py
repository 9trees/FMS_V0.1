import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import math
from skimage.io import imread
import os
from fruit_color import generate_mask
from detect_paper import transform_sheet
from fruit_detection import get_roi

def hsv_slicing(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img)
    h = cv2.GaussianBlur(h,(3,3),0)
    th_val,h_thresh = cv2.threshold(h,0, 255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    cv2.imshow("h_thresh", h_thresh)
    return h_thresh

def get_length(img, cm_to_pixel):
    img = hsv_slicing(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    c = max(contours, key=cv2.contourArea)
    arch_len = cv2.arcLength(c, True)
    veggi_len = arch_len / 2
    veggi_len = round(veggi_len * cm_to_pixel, 2)
    return veggi_len


def midpoint(p1, p2):
    return [(p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2]


def findIntersection(x1, y1, x2, y2, x3, y3, x4, y4):
    px = ((x1 * y2 - y1 * x2) * (x3 - x4) - (x1 - x2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    py = ((x1 * y2 - y1 * x2) * (y3 - y4) - (y1 - y2) * (x3 * y4 - y3 * x4)) / (
            (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4))
    return [px, py]


def get_diameter(img, cm_to_pixel):
    img = hsv_slicing(img)
    contours, hierarchy = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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


if __name__=='__main__':
    IMG_PATH = os.path.join(os.getcwd(), "images", "raw", "1.jpeg") # Input image path
    CHART_PATH = os.path.join(os.getcwd(), "images")# Json Path
    img = cv2.imread(IMG_PATH) #Not CV2 from SKImage
    img = transform_sheet(img)
    scale_len = 2403 - 534 #Scale to pixel Ratio
    cm_to_pixel = 15 / scale_len #15cm Y_Value extraction
    fruits = get_roi(img)
    # Calculate the length, diameter of each fruit
    for i, roi in enumerate(fruits):
        fruit = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        fruit_mask = hsv_slicing(fruit)
        length = get_length(fruit_mask, cm_to_pixel)
        diameter = get_diameter(fruit_mask,cm_to_pixel)
    print(length, diameter)
