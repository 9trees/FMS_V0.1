import os
import cv2

def get_n_roi(img):
    num_fruits = int(input(" Please input the number of fruits in the image! "))
    roi = []
    for i in range(0, num_fruits):
        roi.append(cv2.selectROI(img))
    return roi

def get_roi(img):
    return cv2.selectROI(img)

def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]
    # if both the width and height are None, then return the original image
    if width is None and height is None:
        return image
    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the dimensions
        r = height / float(h)
        dim = (int(w * r), height)
    else:
        # calculate the ratio of the width and construct the dimension
        r = width / float(w)
        dim = (width, int(h * r))
    # resize the image
    resized = cv2.resize(image, dim, interpolation = inter)
    # return the resized image
    return resized
