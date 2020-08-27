import cv2
from matplotlib import pyplot as plt


def get_roi(img):
    fromCenter = False
    ROIs = cv2.selectROIs('Select ROIs', img, fromCenter)
    list_of_rois = []
    while True:

        list_of_rois.append(ROIs)
        if cv2.waitKey(0) & 0xFF == ord('q'):
            cv2.destroyAllWindows()
            break

    list_of_coords = [i.tolist() for i in list_of_rois[0]]

    coords = []
    for i in list_of_coords:
        temp = []
        x, y, w, h = i
        temp.append(x)
        temp.append(y)
        temp.append(x + w)
        temp.append(y + h)
        coords.append(temp)

    return coords


def draw_rois(hsv_show, list_off_cords):
    for i in list_off_cords:
        cv2.rectangle(hsv_show, (i[0], i[1]), (i[2], i[3]), color=(255, 0, 0), thickness=3)
