import cv2
import detect_paper
import matplotlib.pyplot as plt

img_path = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/CHILI_LEARN/FMS_V0.1/tamil/1.jpeg'

sliced_doc = detect_paper.get_a4(img_path)
cm_to_pixel = detect_paper.get_cm_per_pixel(sliced_doc)
