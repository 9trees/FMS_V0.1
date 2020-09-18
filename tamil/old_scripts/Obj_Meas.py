import cv2
import utlis
 
###################################
webcam = False
path = 'IMG_3929.JPG'
cap = cv2.VideoCapture(0)
cap.set(10,160)
cap.set(3,1920)
cap.set(4,1080)
scale = 2
wP = 210 *scale
hP= 297 *scale
###################################
 
while True:
    if webcam:success,img = cap.read()
    else: img = cv2.imread(path)
 
    imgContours , conts = utlis.getContours(img,minArea=50000,filter=4)
    if len(conts) != 0:
        biggest = conts[0][2]
        #print(biggest)
        imgWarp = utlis.warpImg(img, biggest, wP, hP, pad=5)
        imgContours2, conts2 = utlis.getContours(imgWarp, showCanny=True, 
                                                 minArea=5000, filter=0,
                                                 cThr=[75,75],draw = True)
        cv2.imshow('A4_1', imgContours2)
        #print("conts2 {}" .format(conts2))
        if len(conts) != 0:
            for obj in conts2:
                cv2.polylines(imgContours2,[obj[2]],True,(0,255,0),2)
                nPoints = utlis.reorder(obj[2])
                nW = round((utlis.findDis(nPoints[0][0]//scale,nPoints[1][0]//scale)/10),1)
                nH = round((utlis.findDis(nPoints[0][0]//scale,nPoints[2][0]//scale)/10),1)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[1][0][0], nPoints[1][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                cv2.arrowedLine(imgContours2, (nPoints[0][0][0], nPoints[0][0][1]), (nPoints[2][0][0], nPoints[2][0][1]),
                                (255, 0, 255), 3, 8, 0, 0.05)
                x, y, w, h = obj[3]
                cv2.putText(imgContours2, '{}cm'.format(nW), (x + 30, y - 10), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
                cv2.putText(imgContours2, '{}cm'.format(nH), (x - 70, y + h // 2), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5,
                            (255, 0, 255), 2)
        cv2.imshow('A4', imgContours2)
 
    img = cv2.resize(img,(0,0),None,0.5,0.5)
    #cv2.imshow('Original',img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
          break
