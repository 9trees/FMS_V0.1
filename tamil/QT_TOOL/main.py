import cv2
import detect_paper
import matplotlib.pyplot as plt
import get_roi
from skimage.io import imread
import copy
import hsv_slicer
import length_width as lw
import Get_the_color as gc
from pathlib import Path
import pandas as pd

img_path = '/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/CHILI_LEARN/2020-08-20-Okra-fruit/IMG_8733.JPG'

img = imread(img_path)
sliced_doc = detect_paper.get_a4(img)
cm_to_pixel = detect_paper.get_cm_per_pixel(sliced_doc)
hsv_img, hsv_values = hsv_slicer.hsv_slicer(sliced_doc)
list_off_cords = get_roi.get_roi(hsv_img)
hsv_img = cv2.cvtColor(hsv_img, cv2.COLOR_BGR2RGB)
hsv_show = copy.copy(hsv_img)
get_roi.draw_rois(hsv_img, list_off_cords)
animate_img = copy.copy(sliced_doc)

source_dict = {}
img_name = Path(img_path).name.split('.')[0]
source_dict.update({img_name: {}})

count = 1
for coord in list_off_cords:
    crop_img = hsv_show[coord[1]:coord[3], coord[0]:coord[2]]
    length = lw.get_length(crop_img, cm_to_pixel)
    width = lw.get_width(crop_img, cm_to_pixel)
    color = gc.find_color(crop_img)
    lw.animate(animate_img, crop_img, length, width, color,count, coord)
    sample_id = str(count)
    source_dict[img_name].update({sample_id: {
        'length': length,
        'width': width,
        'color': color
    }})
    count += 1

plt.imshow(animate_img)
cv2.imwrite('Animated_'+img_name+'.jpg',cv2.cvtColor(animate_img, cv2.COLOR_RGB2BGR))

samples = []
colors = []
lengths = []
widths = []
for i, j in source_dict[img_name].items():
    samples.append(i)
    colors.append(j['color'])
    lengths.append(j['length'])
    widths.append(j['width'])

source_list = [samples, colors, lengths, widths]

df = pd.DataFrame([source_list])
df.to_csv(img_name + '.csv', index=False, header=False)
