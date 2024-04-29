import cv2
import numpy as np
import os

BasePath = os.path.dirname(__file__) + "\\..\\Images\\"

def ConvertGrayImage(image: np.ndarray,r_ratio,g_ratio,b_ratio) -> np.ndarray:
    b, g, r = cv2.split(image)
    gray_image = np.uint8(r_ratio * r + g_ratio * g + b_ratio * b)
    max_value = np.max(gray_image)
    return gray_image,max_value

def CheckCanny(egde_image: np.ndarray,input_x,input_y,):
    cv2.imshow('Edge Image', egde_image)
    if egde_image[input_x, input_y] == 255:
        print(f"Pixel (x = {input_x}, y = {input_y}) là một điểm biên trên ảnh Ig.")
    else:
        print(f"Pixel (x = {input_x}, y = {input_y}) không phải là một điểm biên trên ảnh Ig.")

def DrawImageWithContour(image: np.ndarray,binary_image: np.ndarray):
    contours, _ = cv2.findContours(binary_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

    max_area = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area:
            max_area = area

    color_yellow = (0, 255, 255)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > max_area / 5.0:
            cv2.drawContours(image, [contour], -1, color_yellow, 3)
    cv2.imshow('Image with Contours', image)


if __name__ == '__main__':
    ImagePath = BasePath + "anh5.jpg"

    #1
    I = cv2.imread(ImagePath,cv2.IMREAD_COLOR)
    cv2.imshow("Input",I)

    #2
    Ig,max_gray_value = ConvertGrayImage(I,0.39,0.5,0.11)
    cv2.imshow("Gray",Ig)

    #3
    Ie = cv2.Canny(Ig,100,200)
    cv2.imshow("Edge Image", Ie)

    #4
    CheckCanny(Ie,326,160)

    #5
    _, Ib = cv2.threshold(Ig, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    cv2.imshow('Binary Image', Ib)

    #6
    DrawImageWithContour(I,Ib)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
