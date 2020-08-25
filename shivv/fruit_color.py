import cv2
import csv
import numpy as np
import os
import json
import numpy as np
from sklearn.cluster import KMeans
from collections import Counter
from detect_paper import transform_sheet

def load_chart(chart_path):
    # Load the color chart as a dictionary
    reader = csv.reader(open(chart_path))
    chart = {}
    for row in reader:
        key = row[1]
        chart[key] = row[0]
    return chart

def color_tag(chart_path, color):
    chart = load_chart(chart_path)
    hue_list = list(chart.keys())
    hue_list = [int(i) for i in hue_list]
    closest_hue = hue_list[min(range(len(hue_list)), key = lambda i: abs(hue_list[i]-color[0]))]
    return chart[str(closest_hue)]

def check_white(color):
    # Define range of white color in HSV
    color_np = np.array(color,dtype=np.uint8 )
    lower_white = np.array([0,0,0], dtype=np.uint8)
    upper_white = np.array([10,10,255], dtype=np.uint8)
    mask = cv2.inRange(color_np, lower_white, upper_white)
    res = cv2.bitwise_and(color_np,color_np, mask= mask)
    res = np.squeeze(res)
    if np.all(res == color_np):
        # print("WHITE")
        return True
    else:
        # print("GREEN")
        return False

def get_color(img, k=2, image_processing_size = None):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # resize image if new dims provided
    if image_processing_size is not None:
        img = cv2.resize(image, image_processing_size,
                            interpolation = cv2.INTER_AREA)
    #reshape the image to be a list of pixels
    img = img.reshape((img.shape[0] * img.shape[1], 3))
    #cluster and assign labels to the pixels
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(img)
    #count labels to find most popular
    label_counts = Counter(labels)
    # dominant_color = clt.cluster_centers_[label_counts.most_common()[0][0]]
    colors_list = [[int(i) for i in colors] for colors in clt.cluster_centers_]
    print(colors_list)
    for i in range(len(colors_list)):
        if check_white(colors_list[i]):
            pass
        else:
            return colors_list[i]
    return "Can't see any dominant_color"


def generate_mask(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img_hsv)
    blur = cv2.GaussianBlur(h,(5,5),0)
    th_val,h_thresh = cv2.threshold(blur,0,np.max(h),cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    #For viewing the thresholding
    h_dash = np.where(h_thresh == 0, 0, h)
    s_dash = np.where(h_dash == 0, 255, s)
    v_dash = np.where(h_dash == 0, 0, v)
    img_thresh = cv2.merge([h_dash, s_dash, v_dash])
    img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_HSV2BGR)
    return img_thresh

if __name__=='__main__':
    IMG_PATH = os.path.join(os.getcwd(), "images", "raw", "1.jpeg") # Input image path
    CHART_PATH = os.path.join(os.getcwd(), "images")# Json Path
    img = cv2.imread(IMG_PATH) #Not CV2 from SKImage
    img = transform_sheet(img)
    img = generate_mask(img)
    while(1):
        cv2.imshow("img", img)
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            break
        elif k==-1:
            continue
