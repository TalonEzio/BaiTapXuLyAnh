import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BasePath = os.path.dirname(__file__) + "\\..\\Images\\"

if __name__ == '__main__':
    ImagePath = BasePath + "anh5.jpg"

    #1
    I = cv2.imread(ImagePath,cv2.IMREAD_COLOR)
    cv2.imshow("Input",I)

    #2
    Ihsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(Ihsv)

    cv2.imshow("v channel",v)
    max_v = np.max(v)
    print("Giá trị mức xám lớn nhất của kênh v:", max_v)

    #3
    kernel = np.ones((5,5),np.float32) / 25
    Is = cv2.filter2D(src=s, ddepth=-1,kernel=kernel)
    cv2.imshow("Blur",Is)

    #4
    reverseIs = 255 - Is
    _,Ib = cv2.threshold(reverseIs,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("binary",Ib)

    #5
    contours, _ = cv2.findContours(Ib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    result = cv2.imread(ImagePath)
    cv2.drawContours(result, [max_contour], -1, (0, 255, 0), 2)

    cv2.imshow("Contour", result)


    #6

    # Tăng độ sáng kênh V bằng phương pháp giãn mức xám
    adjusted_v = cv2.equalizeHist(v)
    adjusted_hsv = Ihsv.copy()
    adjusted_hsv[:,:,2] = adjusted_v

    cv2.imshow("Adjusted Image", cv2.cvtColor(adjusted_hsv, cv2.COLOR_HSV2BGR))

    cv2.waitKey(0)
    cv2.destroyAllWindows()
