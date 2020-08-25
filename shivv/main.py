import os
import csv
import cv2
import numpy as np
from detect_paper import transform_sheet
from fruit_dimensions import get_length, get_diameter
from fruit_detection import get_roi, image_resize
from fruit_color import get_color, generate_mask, color_tag


########## INPUTS ###########
IMG_PATH = os.path.join(os.getcwd(), "images", "raw", "1.jpeg") # Input image path
CHART_PATH = os.path.join(os.getcwd(), "images", "charts", "okra-chart.csv")              # Json Path

########## OUTPUTS ###########
fruit_lengths = []
fruit_diameters = []
fruit_colors = []
fruit_ids = []

def export_data(csv_filename, list):
    with open(csv_filename, "a") as fp:
        wr = csv.writer(fp, dialect='excel')
        wr.writerow(list)
    return 1

def animate_results(img, fruit_number, roi, length, diameter, color, color_name):
    fontScale = (img.shape[0] * img.shape[1]) / (1250 * 1250)
    text_color = (255,255,255)
    # Calculating the color for drawing the boxes
    color_np = np.array(color, dtype=np.uint8)
    color_np = color_np.reshape(1,1,3)
    color = cv2.cvtColor(color_np, cv2.COLOR_HSV2BGR)
    color = np.squeeze(color)
    color = (int(color[0]), int(color[1]), int(color[2]))

    # Box the object
    cv2.rectangle(img , (roi[0], roi[1]), (roi[0]+roi[2], roi[1]+roi[3]), (color), 2)
    cv2.rectangle(img, (roi[0], roi[1]+roi[3]), (roi[0]+roi[2],roi[1]+roi[3]+int(200*fontScale)), color, cv2.FILLED)
    # Number to object
    cv2.putText(img, str(fruit_number) ,(int(roi[0]+roi[2]/2), int(roi[1]+roi[3]/2)) , cv2.FONT_HERSHEY_DUPLEX, 2*fontScale , text_color, 1)
    # Write the Color and length
    cv2.putText(img,"C: " + str(color_name),(roi[0], roi[1]+roi[3]+ int(50 * fontScale)) , cv2.FONT_HERSHEY_DUPLEX, 1.1*fontScale, text_color, 1)
    cv2.putText(img,"L: " + str(length) + " cm",(roi[0]+1, roi[1]+roi[3]+int(100*fontScale)) , cv2.FONT_HERSHEY_DUPLEX, 1.1*fontScale , text_color, 1)
    cv2.putText(img,"D: " + str(diameter) + " cm",(roi[0]+1, roi[1]+roi[3]+int(150*fontScale)) , cv2.FONT_HERSHEY_DUPLEX, 1.1*fontScale , text_color, 1)
    # cv2.putText(img,"Hue : " + str(color_hsv[0][0][0]),(roi[0], roi[1]+roi[3]+ int(150 * fontScale)) , cv2.FONT_HERSHEY_DUPLEX,  fontScale, text_color, 1)
    return img


def main(img_path, chart_path):
    img = cv2.imread(img_path)
    # Detect paper
    img = transform_sheet(img)
    img = image_resize(img, width = 800)
    # Calculate the scale factor
    max_dimension = max(img.shape[0], img.shape[1])
    cm_to_pixel = 29.7 / max_dimension # A4 sheet size = 29.7 cm x 21 cm
    # Get ROIs dynamically
    fruit_number = 0
    while(1):
        roi = get_roi(img)
        fruit_number+=1
        fruit = img[int(roi[1]):int(roi[1]+roi[3]), int(roi[0]):int(roi[0]+roi[2])]
        fruit_mask = generate_mask(fruit)
        cv2.imshow("mask", fruit_mask)
        # Get the color value
        color = get_color(fruit_mask)
        # Convert the color to a tag
        color_name = color_tag(CHART_PATH, color)
        # Get the length of the fruit
        length = (get_length(fruit_mask, cm_to_pixel))
        # Get the diameter of the fruit
        diameter = get_diameter(fruit_mask, cm_to_pixel)
        # Exporting the results to csv
        fruit_ids.append(fruit_number)
        fruit_colors.append(color_name)
        fruit_lengths.append(length)
        fruit_diameters.append(diameter)
        img_out = animate_results(img, fruit_number, roi, length, diameter, color, color_name)
        k = cv2.waitKey(33)
        if k==27:    # Esc key to stop
            break
        elif k==-1:  # normally -1 returned,so don't print it
            continue
    out_file = IMG_PATH.split('.')
    cv2.imwrite(IMG_PATH.split('.')[0] + "-out." + IMG_PATH.split('.')[1],img_out)
    export_data(IMG_PATH.split('.')[0] + ".csv", [fruit_ids, fruit_colors, fruit_lengths, fruit_diameters] )
    return 1

if __name__ == '__main__' :
    main(IMG_PATH, CHART_PATH)
