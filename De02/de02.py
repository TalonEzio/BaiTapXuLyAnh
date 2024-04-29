import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BasePath = os.path.dirname(__file__) + "\\..\\Images\\"

if __name__ == '__main__':
    ImagePath = BasePath + "anh5.jpg"

    #1
    I = cv2.imread(ImagePath,cv2.IMREAD_COLOR)
    #cv2.imshow("Input",I)

    b,_,_ = cv2.split(I)
    print("Kênh b: ")
    print(b)

    #2
    Ihsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(Ihsv)

    cv2.imshow("h channel",h)


    mean_s = np.mean(s)
    print("Giá trị trung bình của kênh S:", mean_s)

    #3
    hist_s = cv2.calcHist([Ihsv], [1], None, [256], [0, 256])

    plt.plot(hist_s, color='red')
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Số lượng pixel')
    plt.title('Histogram kênh S')
    #plt.show()

    #4
    ksize = (3, 3)
    Is = cv2.blur(v, ksize)
    cv2.imshow("Blur",Is)

    #5
    _,Ib = cv2.threshold(Is, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow("Binary",Ib)

    #6, chưa làm
    contours,_ = cv2.findContours(Ib, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(I, contours, -1, (0, 0, 255), 2)

    cv2.imshow('Contour', I)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
