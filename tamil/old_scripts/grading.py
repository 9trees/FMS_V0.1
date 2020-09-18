import cv2
import numpy as np
import os
import json

from sklearn.cluster import KMeans
from collections import Counter
import csv

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

def export_data(list, file_name):
    file_name = file_name + ".csv"
    with open(file_name, "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(list)
    return 1

def check_white(color):
    # Define range of white color in HSV
    lower_white = [150,150,150]
    upper_white = [255,255,255]
    # Create the mask
    if lower_white[:] < color[:]:
        return True
    else:
        return False

def get_dominant_color(image, k=2, image_processing_size = None):
    # resize image if new dims provided
    if image_processing_size is not None:
        image = cv2.resize(image, image_processing_size,
                            interpolation = cv2.INTER_AREA)

    #reshape the image to be a list of pixels
    image = image.reshape((image.shape[0] * image.shape[1], 3))

    #cluster and assign labels to the pixels
    clt = KMeans(n_clusters = k)
    labels = clt.fit_predict(image)
    #count labels to find most popular
    label_counts = Counter(labels)
    #subset out most popular centroid
    dominant_color = clt.cluster_centers_[label_counts.most_common()[0][0]]
    dominant_color_list = [int(i) for i in dominant_color]
    is_white = True
    if is_white == check_white(dominant_color_list):
        dominant_color = clt.cluster_centers_[label_counts.most_common()[1][0]]
    dominant_color_list = [int(i) for i in dominant_color]
    return dominant_color_list


def get_color(img):
    img_hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(img_hsv)
    blur = cv2.GaussianBlur(h,(5,5),0)
    th_val,h_thresh = cv2.threshold(blur,0,np.max(h),cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    h_cal = np.minimum(h, h_thresh)

    #For viewing the thresholding
    h_max =  np.max(h_cal)
    h_dash = np.where(h_cal == h_max, h_max, h)
    s_dash = np.where(h_dash == h_max, 0, s)
    v_dash = np.where(h_dash == h_max, 255, v)
    img_thresh = cv2.merge([h_dash, s_dash, v_dash])
    img_thresh = cv2.cvtColor(img_thresh, cv2.COLOR_HSV2BGR)

    # Getting dominant color value
    color_bgr= get_dominant_color(img_thresh, k=2, image_processing_size = None)
    return color_bgr



if __name__ == '__main__' :
    # Read image
    
    filename = "IMG_8671.JPG"
    dir = os.path.join(os.getcwd(), "images")
    img = cv2.imread(os.path.join(dir, filename))
    img_out = img = im_resize(img, scale_percent=40)
    print(" Please select a rectangle with height as 1cm")
    #cv2.namedWindow("Image_ROI",2)
    scale_roi = cv2.selectROI(img)
    scale_pixels = scale_roi[3]
    fontScale = (img.shape[0] * img.shape[1]) / (3000 * 3000)
    print(int(40 * fontScale))
    # Select ROI
    for i in range(0,25):
        roi = cv2.selectROI(img)
        img_crop = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        text_color = (255,255,255)
        color = get_color(img_crop)
        color_np = np.uint8([[[color[0],color[1],color[2]]]])
        color_hsv = cv2.cvtColor(color_np, cv2.COLOR_BGR2HSV)
        length = round((roi[3]/scale_pixels),2)
        # Box the object
        cv2.rectangle(img_out , (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]),color, 2)
        cv2.rectangle(img_out, (roi[0], roi[1]+roi[3]), (roi[0]+roi[2],roi[1]+roi[3]+int(200*fontScale)),color,cv2.FILLED)
        # Number to object
        cv2.putText(img_out, str(i+1) ,(int(roi[0]+roi[2]/2), int(roi[1]+roi[3]/2)) , cv2.FONT_HERSHEY_DUPLEX, 2*fontScale , text_color, 1)
        # Write the Color and length
        cv2.putText(img_out,str(length*10) + " mm",(roi[0]+1, roi[1]+roi[3]+int(50*fontScale)) , cv2.FONT_HERSHEY_DUPLEX, fontScale , text_color, 1)
        cv2.putText(img_out,str(color),(roi[0], roi[1]+roi[3]+ int(100 * fontScale)) , cv2.FONT_HERSHEY_DUPLEX,  fontScale, text_color, 1)
        cv2.putText(img_out,"Hue : " + str(color_hsv[0][0][0]),(roi[0], roi[1]+roi[3]+ int(150 * fontScale)) , cv2.FONT_HERSHEY_DUPLEX,  fontScale, text_color, 1)
        # Save Image and CSV data
        out_path = dir + filename.split(".")[0] + "-out.jpeg"
        data_list = [i+1 ,length*10, color[2], color[1], color[0]]
        export_data(data_list, filename.split(".")[0])
        cv2.imwrite(out_path, img_out)
        
