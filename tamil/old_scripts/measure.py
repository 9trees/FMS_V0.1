#!/usr/bin/python
######################################################################################################                                                                                                
#                  `...          ..-`                                                                #
#                  -cpy`        `cpy:                                                                #
#        ```.```   -oo+``````    cpy-    ```.``     ```` ```   ``````` ``.``...``  ````    ````      #
#      `-/+ocpy+-` -cpy/+cpy+-   cpy-  `:/+cpy+/-`  :oo//+o: `:cpy+o+/-/sy+ocpyso- .cpy-   +yy/      #
#     `:oo+.../++-`-cpy:..:cpy`  cpy- `cpy/.../oo/` :cpy+:-..cpy:..-cpy:-ys-../cpy. :yys` .yys`      #
#     `cpy.    ``  -cpy`  `cpy.  cpy- .cpy`   .cpy. :oo+`   :oo+`   -cpy`s:   `cpy/ `+yy/`cpy-       #
#     `cpy-   ...` -cpy`  `cpy.  cpy- .cpy.   -cpy` :oo/`   -cpy`   -oo+`y/   .cpy-  `sys/yy/        #
#      -cpy/:/oo+.`-cpy`  `cpy. `cpy-  :cpy/:/oo+-  :oo/`   `:oo+::/oo+:cpy+/+cpy/`   -cpyyo`        #
#       `-:///:-`  .:::`  `:::`  :::.   `-:///:-`   -::-     `.-////:-`cpy+/+++/.      /cpy.         #
#                                                                     `cpy/         .//cpy:          #
#                                                                      /++-         -cpy+.           #
######################################################################################################
__author__ = "Sivashankar Palraj and Apoorv Vaish"
__copyright__ = "Copyright 2020, The Cogent Project"
__credits__ = ["Sivashankar Palraj", "Apoorv Vaish", "Gajendra Babu B., Ph. D"]
__license__ = "copyright © ChloroPy, Singapore"
__version__ = "0.0.1"
__date__ = "03-08-2020"
__maintainer__ = "Sivashankar Palraj"
__email__ = "sivashankaryadav@outlook.com"
__status__ = "Production"

import cv2
import numpy as np
import os

def im_resize(img, scale_percent = 60):
    """
    The function resize the image by Up/Down Scale.
    Image and Scale % are the input and Resized Img is Output.
    """
    print('Original Dimensions : ',img.shape)
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    print('Resized Dimensions : ',resized.shape)

    return resized



def getMeasure(img, showCanny=False):
    """
    The function returns Length and width of the object in Image
    """
    """ #Converting BGR to Gaay scale Image.
    imgGray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    #Since edge detection is susceptible to noise in the image, first step is to remove the noise in the image with a 5x5 Gaussian filter.
    imgBlur = cv2.GaussianBlur(imgGray, (5,5), cv2.BORDER_DEFAULT) """
    imgHsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_Green = np.array([35, 37, 0])
    upper_Green = np.array([105, 255, 255])
    mask = cv2.inRange(imgHsv, lower_Green, upper_Green)
    imgBlur = cv2.GaussianBlur(mask, (5,5), cv2.BORDER_DEFAULT)
    #Auto Canny-Edge Detection
    imgCanny = cv2.Canny(imgBlur, 100, 100)
    kernel = np.ones((5,5))
    imgDial = cv2.dilate(imgCanny,kernel,iterations=3)
    imgThre = cv2.erode(imgDial,kernel,iterations=2)
    if showCanny:cv2.imshow('Canny',imgThre)
    #contours,hiearchy = cv2.findContours(imgThre,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    #return contours

def Scale(img):
    """
    The function returns a pixel value with respect to 1cm
    """
    print(" Please select a square with sides as 1cm")
    scale_roi = cv2.selectROI(img)
    scale_pixels = scale_roi[3]
    return scale_pixels


def main():
    filename = "IMG_8674.JPG"
    fdir = os.path.join("C:\\Users\\USER\\Desktop\\ChloroPy", "images")
    img = cv2.imread(os.path.join(fdir, filename))
    img_out = img = im_resize(img, scale_percent=40)
    #scale_pixels = Scale(img)
    for i in range(1, 11):
        roi = cv2.selectROI(img)
        img_crop = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        
        contours = getMeasure(img_crop, showCanny=True)
        print(i, contours)


if __name__ == "__main__":
    main()