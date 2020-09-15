import cv2
import numpy as np
from skimage.io import imread
from pathlib import Path
import json
import os
import base64


def image_data_dump(imagePath):
    with open(imagePath, 'rb') as f:
        imageData = f.read()
        imageData = base64.b64encode(imageData).decode('utf-8')
    return imageData


def save_annotation(contours, label_name, img, img_path, img_name, out):
    coords = contours
    coords = [np.concatenate(i) for i in coords]
    coords = [i.tolist() for i in coords]

    height, width, _ = img.shape

    imageData = image_data_dump(img_path)

    # this is section is for one class

    shapes = []
    for i in coords:
        shapes.append(
            {"shape_type": 'polygon', "line_color": None, "points": i, "fill_color": None, "label": label_name})

    myDictObj = {"imagePath": img_name, "imageData": imageData, "shapes": shapes,
                 "version": "3.11.2", "flags": {}, "fillColor": [255, 0, 0, 128],
                 "lineColor": [0, 255, 0, 128], "imageWidth": width, "imageHeight": height}

    # when you need this for more class use this below
    # shapes = []
    # for i in range(len(coords)):
    #     for cnt in coords[i]:
    #         shapes.append(
    #             {"shape_type": 'polygon', "line_color": None, "points": cnt, "fill_color": None, "label": str(i)})
    #
    # myDictObj = {"imagePath": image_name, "imageData": imageData, "shapes": shapes,
    #              "version": "3.11.2", "flags": {}, "fillColor": [255, 0, 0, 128],
    #              "lineColor": [0, 255, 0, 128], "imageWidth": width, "imageHeight": height}
    #

    with open(out, 'w', encoding='utf-8') as f:
        json.dump(myDictObj, f, ensure_ascii=False, indent=2)
