import cv2
import numpy as np
import matplotlib.pyplot as plt
from shapely.geometry import LineString
import math
from shapely.geometry import Polygon

img_path = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/CHILI_LEARN/Okra/1.jpg'
from skimage.io import imread

img = imread(img_path)


# x1, y1 = 2616, 1265
# x2, y2 = 2980, 1908
# img = img[y1:y2, x1:x2]
#
# scale_len = 2403 - 534
# cm_to_pixel = 15 / scale_len


def hsv_slicing(img):
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, (26, 52, 0), (65, 255, 255))
    imask = mask > 0
    green = np.zeros_like(img, np.uint8)
    green[imask] = img[imask]
    gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
    ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return threshold


def get_length(c, cm_to_pixel):
    # contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # c = max(contours, key=cv2.contourArea)
    # arch_len = cv2.arcLength(c, True)

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


def get_width(c, cm_to_pixel):
    # contours, _ = cv2.findContours(img, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
    # c = max(contours, key=cv2.contourArea)
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


# binery_thresh = hsv_slicing(img)
# length = get_length(binery_thresh, cm_to_pixel)
# width = get_width(binery_thresh, cm_to_pixel)
# max_width = get_max_width(binery_thresh, cm_to_pixel)

img_path = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/CHILI_LEARN/Okra/1.jpg'
from skimage.io import imread

img = imread(img_path)
binery_thresh = hsv_slicing(img)
kernel = np.ones((5, 5), np.uint8)
dilation = cv2.dilate(binery_thresh, kernel, iterations=2)
# plt.imshow(dilation)

contours, _ = cv2.findContours(binery_thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
area = [cv2.contourArea(i) for i in contours]

sliced_cnts = []
for cnt in contours:
    if 200000 < cv2.contourArea(cnt):
        sliced_cnts.append(cnt)
cm_to_pixel = 15.1/3048

contour_dict = {}
for i in sliced_cnts:
    contour_bound = i.tolist()
    contour_bound = [i[0] for i in contour_bound]
    P = Polygon(contour_bound)
    point = list(P.centroid.coords)[0]
    length = get_length(i,cm_to_pixel)
    # length =
    # contour_dict.update({point: i})
    contour_dict.update({point:{'bound':i,'length':length}})

contour_dict = sorted(contour_dict.items(), key=lambda x: x[0][0])

count = 1
for i in contour_dict:
    font = cv2.FONT_HERSHEY_SIMPLEX
    bottomLeftCornerOfText = (10, 500)
    fontScale = 2
    fontColor = (255, 255, 255)
    text = str(count)+'- '+'length: '+str(i[1]['length'])

    cv2.putText(img,text ,
                (int(i[0][0]), int(i[0][1])),
                font,
                fontScale,
                fontColor,10)
    count += 1
plt.imshow(img)























# ===========================================================
# x1, y1 = 1043, 270
# x2, y2 = 1754, 920
# crop_img = img[y1:y2, x1:x2]
#
#
# scale_len = 2403 - 534
# cm_to_pixel = 15 / scale_len
#
# hsv = cv2.cvtColor(crop_img, cv2.COLOR_RGB2HSV)
# mask = cv2.inRange(hsv, (0, 0, 0), (255, 25, 147))
# imask = mask > 0
# green = np.zeros_like(crop_img, np.uint8)
# green[imask] = crop_img[imask]
# gray = cv2.cvtColor(green, cv2.COLOR_BGR2GRAY)
# ret, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
# kernel = np.ones((5, 5), np.uint8)
# dilation = cv2.dilate(threshold, kernel, iterations=3)
# kernel1 = np.ones((10, 10), np.uint8)
# erosion = cv2.erode(dilation,kernel1,iterations=2)
# kernel = np.ones((10, 10), np.uint8)
# dilation = cv2.dilate(erosion, kernel, iterations=3)
# plt.imshow(dilation)
# contours, _ = cv2.findContours(dilation, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
# c = max(contours, key=cv2.contourArea)
# area_coin = cv2.contourArea(c)
# coin_area_cm = 3.801
# cm_to_pixel = coin_area_cm/area_coin
#
# cm_to_pixel = 15.1/3048