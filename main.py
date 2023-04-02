import cv2
import numpy as np

# Define the Roberts filter function
def my_roberts(slika):
    x = np.array([[1, 0], [0, -1]])
    y = np.array([[0, 1], [-1, 0]])

    slika = slika.astype(np.float32)

    edgeX = cv2.filter2D(slika, -1, x)
    edgeY = cv2.filter2D(slika, -1, y)

    edges_mag = np.sqrt(np.square(edgeX) + np.square(edgeY))

    slika_robov = edges_mag.astype(np.uint8)

    return slika_robov

def my_prewitt(slika):
    x = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]])
    y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]])

    slika = slika.astype(np.float32)

    edgeX = cv2.filter2D(slika, -1, x)
    edgeY = cv2.filter2D(slika, -1, y)

    edges_mag = np.sqrt(np.square(edgeX) + np.square(edgeY))

    slika_robov = edges_mag.astype(np.uint8)

    return slika_robov

def my_sobel(slika):
    x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
    y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])

    slika = slika.astype(np.float32)

    edgeX = cv2.filter2D(slika, -1, x)
    edgeY = cv2.filter2D(slika, -1, y)

    edges_mag = np.sqrt(np.square(edgeX) + np.square(edgeY))

    slika_robov = edges_mag.astype(np.uint8)

    return slika_robov

img = cv2.imread("images/Slika_1.jpg", cv2.IMREAD_GRAYSCALE)

img_roberts = my_roberts(img)
img_prewitt = my_prewitt(img)
img_sobel = my_sobel(img)

cv2.imwrite("images/roberts.jpg", img_roberts)
cv2.imwrite("images/prewitt.jpg", img_prewitt)
cv2.imwrite("images/sobel.jpg", img_sobel)
