import cv2

img = cv2.imread('/mnt/dash/Alpha_Share/Automation_Team/Tamil/NLP_learning/CHILI_LEARN/FMS_V0.1/tamil/1.jpeg')

fromCenter = False
ROIs = cv2.selectROIs('Select ROIs', img, fromCenter)

ROI_1 = img[ROIs[0][1]:ROIs[0][1]+ROIs[0][3], ROIs[0][0]:ROIs[0][0]+ROIs[0][2]]
ROI_2 = img[ROIs[1][1]:ROIs[1][1]+ROIs[1][3], ROIs[1][0]:ROIs[1][0]+ROIs[1][2]]
ROI_3 = img[ROIs[2][1]:ROIs[2][1]+ROIs[2][3], ROIs[2][0]:ROIs[2][0]+ROIs[2][2]]

cv2.imshow('1', ROI_1)
cv2.imshow('2', ROI_2)
cv2.imshow('3', ROI_3)

cv2.waitKey(0)
cv2.destroyAllWindows()

