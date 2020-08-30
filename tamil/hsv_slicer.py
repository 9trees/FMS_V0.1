import cv2
import numpy as np



def nothing(x):
  pass

def hsv_slicer(img):
   cv2.namedWindow('Chloropy')


   cv2.createTrackbar("R1", "Chloropy",0,255,nothing)
   cv2.createTrackbar("G1", "Chloropy",0,255,nothing)
   cv2.createTrackbar("B1", "Chloropy",0,255,nothing)
   cv2.createTrackbar("R2", "Chloropy",255,255,nothing)
   cv2.createTrackbar("G2", "Chloropy",255,255,nothing)
   cv2.createTrackbar("B2", "Chloropy",255,255,nothing)


   rimg = cv2.resize(img, (0,0), fx=1, fy=1)



   while(1):

      R1=cv2.getTrackbarPos("R1", "Chloropy")
      G1=cv2.getTrackbarPos("G1", "Chloropy")
      B1 = cv2.getTrackbarPos("B1", "Chloropy")
      R2 = cv2.getTrackbarPos("R2", "Chloropy")
      G2 = cv2.getTrackbarPos("G2", "Chloropy")
      B2 = cv2.getTrackbarPos("B2", "Chloropy")
      hsv = cv2.cvtColor(rimg, cv2.COLOR_RGB2HSV)
      mask = cv2.inRange(hsv, (R1, G1, B1), (R2, G2, B2))
      imask = mask > 0
      green = np.zeros_like(rimg, np.uint8)
      green[imask] = rimg[imask]
      final = cv2.cvtColor(green, cv2.COLOR_RGB2BGR)

      cv2.imshow("Slicing Tool", final)

      if cv2.waitKey(1) & 0xFF == ord('s'):
         val = [[R1, G1, B1], [R2, G2, B2]]
         cv2.destroyAllWindows()
         break


   return final,val
