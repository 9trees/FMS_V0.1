import cv2
import detect_paper
import matplotlib.pyplot as plt
import get_roi
from skimage.io import imread
import copy
import hsv_slicer

img_path = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/CHILI_LEARN/FMS_V0.1/tamil/1.jpeg'

img = imread(img_path)
sliced_doc = detect_paper.get_a4(img)
cm_to_pixel = detect_paper.get_cm_per_pixel(sliced_doc)
list_off_cords = get_roi.get_roi(cv2.cvtColor(sliced_doc, cv2.COLOR_RGB2BGR))
hsv_show = copy.copy(sliced_doc)
get_roi.draw_rois(hsv_show,list_off_cords)
hsv_values = hsv_slicer.hsv_slicer(hsv_show)