import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

BasePath = os.path.dirname(__file__) + "\\..\\Images\\"

if __name__ == '__main__':
    ImagePath = BasePath + "anh5.jpg"

    #1
    I = cv2.imread(ImagePath,cv2.IMREAD_COLOR)
    #cv2.imshow("Image",I)

    height, width,_ = I.shape
    print(f"tỷ lệ width / height = {width / height}")
    #2

    new_width = 256
    new_height = int(height * (new_width / float(width)))

    I2 = cv2.resize(I, (new_width, new_height), interpolation=cv2.INTER_AREA)

    #cv2.imshow("Resize Image", I2)

    #3
    Ihsv = cv2.cvtColor(I,cv2.COLOR_BGR2HSV)
    h,s,v = cv2.split(Ihsv)

    #cv2.imshow("S channel",s)

    #4
    s_blur = cv2.medianBlur(s,3)

    Ihsv[:,:,1] = s_blur

    I3 = cv2.cvtColor(Ihsv,cv2.COLOR_HSV2BGR)

    #cv2.imshow("Raw after blur",I3)

    #5
    hist_s = cv2.calcHist([Ihsv], [1], None, [256], [0, 256])

    plt.plot(hist_s, color='red')
    plt.xlabel('Giá trị pixel')
    plt.ylabel('Số lượng pixel')
    plt.title('Histogram kênh S')
    plt.show()

    #6

    adjusted_s = cv2.equalizeHist(s)
    adjusted_hsv = Ihsv.copy()
    adjusted_hsv[:,:,2] = adjusted_s

    I4 = cv2.cvtColor(adjusted_hsv,cv2.COLOR_HSV2BGR)

    cv2.imshow("adjusted hsv",I4)

    cv2.waitKey(0)
    cv2.destroyAllWindows()
