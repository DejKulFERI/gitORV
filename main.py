import cv2
import numpy as np
import math
from PIL import Image, ImageFilter
import tkinter as tk
from tkinter import *

def convolve2D(image, kernel):
    rows, cols = image.shape
    krows, kcols = kernel.shape

    output = np.zeros_like(image)

    for i in range(rows - krows + 1):
        for j in range(cols - kcols + 1):
            region = image[i:i+krows, j:j+kcols]
            dot_product = np.sum(region * kernel)
            output[i+krows//2, j+kcols//2] = dot_product

    return output


#Pomoc implementacije od: https://www.geeksforgeeks.org/python-opencv-roberts-edge-detection/
def my_roberts(slika):
    x = np.array([[1, 0], [0, -1]])
    y = np.array([[0, 1], [-1, 0]])

    slika = slika.astype(np.float32)
    slika /= 255.0

    edgeX = convolve2D(slika, x)
    edgeY = convolve2D(slika, y)

    edges_mag = np.sqrt(np.square(edgeX) + np.square(edgeY))
    edges_mag *= 255
    
    slika_robov = edges_mag.astype(np.uint8)
    return slika_robov

def my_prewitt(slika):
    x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    slika = slika.astype(np.float32)

    edgeX = convolve2D(slika, x)
    edgeY = convolve2D(slika, y)

    edges_mag = np.sqrt(np.square(edgeX) + np.square(edgeY))

    slika_robov = edges_mag.astype(np.uint8)

    return slika_robov

def my_sobel(slika):
    x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    slika = slika.astype(np.float32)

    edgeX = convolve2D(slika, x)
    edgeY = convolve2D(slika, y)

    edges_mag = np.sqrt(np.square(edgeX) + np.square(edgeY))

    slika_robov = edges_mag.astype(np.uint8)

    return slika_robov

#Pomagal s pomocjo: https://java2blog.com/cv2-canny-python/
def canny(slika, sp_prag, zg_prag):
    slika_robov = cv2.Canny(slika, zg_prag, sp_prag)
    return slika_robov 


def spremeni_kontrast(slika, alfa, beta):
    slika = np.multiply(slika, alfa)
    slika = np.add(slika, beta)
    return slika

img = cv2.imread("images/Slika_1.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Navadna", img)
img_canny2 = canny(img, 100, 50)
img = cv2.GaussianBlur(img,(5,5),0)

img = spremeni_kontrast(img, 1, 50)
cv2.imshow("Spremenjena", img)

img_roberts = my_roberts(img)
img_prewitt = my_prewitt(img)
img_sobel = my_sobel(img)
img_canny = canny(img, 100, 50)

cv2.imshow("Roberts", img_roberts)
cv2.imshow("Prewitt", img_prewitt)
cv2.imshow("Sobel", img_sobel)
cv2.imshow("Canny", img_canny)

cv2.waitKey(0)

cv2.destroyAllWindows()
