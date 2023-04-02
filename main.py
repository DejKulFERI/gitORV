import cv2
import numpy as np

def my_roberts(slika):
    x = np.array([[1, 0], [0, -1]])
    y = np.array([[0, 1], [-1, 0]])

    slika = slika.astype(np.float32)

    edgeX = cv2.filter2D(slika, -1, x)
    edgeY = cv2.filter2D(slika, -1, y)

    edges_mag = np.sqrt(np.square(edgeX) + np.square(edgeY))
    slika_robov = edges_mag.astype(np.uint8)

    return slika_robov

img = cv2.imread("images/Slika_1.jpg", cv2.IMREAD_GRAYSCALE)

img_roberts = my_roberts(img)

cv2.imwrite("images/roberts.jpg", img_roberts)
