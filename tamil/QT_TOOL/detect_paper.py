import cv2
import utlis


def get_a4(img):
    scale = 3
    wP = 210 *scale
    hP= 297 *scale
    imgContours, conts = utlis.getContours(img, minArea=50000, filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        imgWarp = utlis.warpImg(img, biggest, hP, wP, pad=5)

        return imgWarp

    else:
        print('Not able to detect the A4 sheet')
        return None

def get_cm_per_pixel(sliced_doc):
    h_global, w_global = 21.0, 29.7
    h_local, w_local, _ = sliced_doc.shape
    cm_per_pixel = h_global/h_local
    return cm_per_pixel
