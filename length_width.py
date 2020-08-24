import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import math
import os

filename = "IMG_8671.JPG"
fdir = os.path.join("C:\\Users\\USER\\Desktop\\ChloroPy", "images")
#img = cv2.imread(os.path.join(fdir, filename))
# = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/CHILI_LEARN/IMG_8671.JPG'
from skimage.io import imread

img = imread(os.path.join(fdir, filename)) #Not CV2 from SKImage
x1, y1 = 2616, 1265 #ROI point
x2, y2 = 2980, 1908 #ROI Point
img = img[y1:y2, x1:x2]

scale_len = 2403 - 534 #Scal to pixel Ratio
cm_to_pixel = 15 / scale_len #15cm Y_Value extraction


def hsv_slicing(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV) #RGB to HSV
    mask = cv2.inRange(hsv, (26, 52, 0), (65, 255, 255)) #Upper and lower
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold


def get_length(img, cm_to_pixel):
    _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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


def get_width(img, cm_to_pixel):
    _, contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
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


binery_thresh = hsv_slicing(img)
plt.imshow(binery_thresh)
length = get_length(binery_thresh, cm_to_pixel)
width = get_width(binery_thresh,cm_to_pixel)
print(length, width)
plt.show()
