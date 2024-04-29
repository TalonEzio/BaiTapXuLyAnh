import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BasePath = os.path.dirname(__file__) + "\\..\\Images\\"

if __name__ == '__main__':
    ImagePath = BasePath + "anh5.jpg"

    #1
    I = cv2.imread(ImagePath,cv2.IMREAD_COLOR)
    cv2.imshow("Image",I)

    #2

    Ihsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(Ihsv)
    cv2.imshow("h channel",h)

    max_s = np.max(s)

    print(f'giá trị mức sáng lớn nhất kênh S: {max_s}')

    #3

    hist_v = cv2.calcHist([Ihsv], [2], None, [256], [0, 256])

    plt.plot(hist_v, color='red')
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Số lượng pixel')
    plt.title('Histogram kênh S')
    plt.show()

    #4

    Is = cv2.medianBlur(s,7)
    cv2.imshow("Is",Is)

    #5
    reverse_Is = 255 - Is
    _,Ib = cv2.threshold(reverse_Is,127,255,cv2.THRESH_BINARY)
    cv2.imshow("Ib",Ib)

    contours, _ = cv2.findContours(Ib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    max_contour = None
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area
            max_contour = contour

    result = I.copy()

    for contour in contours:
        cv2.drawContours(result, [contour], -1, (0, 255, 0), 2)

    cv2.imshow("Result",result)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
