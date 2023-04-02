import cv2
import numpy as np
import math
import tkinter as tk
from tkinter import *

def convolve2D(image, kernel):
    # Get the dimensions of the image and the kernel
    rows, cols = image.shape
    krows, kcols = kernel.shape

    # Define the output image
    output = np.zeros_like(image)

    # Loop over the image pixels
    for i in range(rows - krows + 1):
        for j in range(cols - kcols + 1):
            # Extract the kernel-sized region from the image
            region = image[i:i+krows, j:j+kcols]
            # Compute the dot product between the region and the kernel
            dot_product = np.sum(region * kernel)
            # Store the result in the output image
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

    # Convert the output image to the CV_8U data type
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

# Read the input image
img = cv2.imread("images/Slika_2.jpg", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Navadna", img)

def spremeni_kontrast(slika, alfa, beta):
    slika = np.multiply(slika, alfa)
    slika = np.add(slika, beta)
    return slika

def update_image():
    # Get the current values of the sliders
    alfa = alpha_slider.get()
    beta = beta_slider.get()

    # Apply the spremeni_kontrast function to the image with the current slider values
    new_img = spremeni_kontrast(img, alfa, beta)

    # Update the image in the window
    cv2.imshow("Image", new_img)

# Create the tkinter window
root = tk.Tk()
root.title("Image Viewer")

# Create a frame to hold the sliders
slider_frame = tk.Frame(root)
slider_frame.pack(side=tk.TOP, fill=tk.BOTH, padx=5, pady=5)

# Create a slider for the alpha value
alpha_slider = tk.Scale(slider_frame, from_=0.1, to=3.0, resolution=0.1, label="Alpha", orient=tk.HORIZONTAL, length=200, command=update_image)
alpha_slider.set(1.0)
alpha_slider.pack(side=tk.LEFT, padx=5)

# Create a slider for the beta value
beta_slider = tk.Scale(slider_frame, from_=-100.0, to=100.0, resolution=1.0, label="Beta", orient=tk.HORIZONTAL, length=200, command=update_image)
beta_slider.set(0.0)
beta_slider.pack(side=tk.LEFT, padx=5)

# Apply the Roberts filter to the image
#img_roberts = my_roberts(img)
#img_prewitt = my_prewitt(img)
#img_sobel = my_sobel(img)
#img_canny = canny(img, 100, 50)

#cv2.imshow("Roberts", img_roberts)
#cv2.imshow("Prewitt", img_prewitt)
#cv2.imshow("Sobel", img_sobel)
#cv2.imshow("Canny", img_canny)
  
apply_button = tk.Button(root, text="Apply", command=update_image)
apply_button.pack(side=tk.TOP, padx=5, pady=5)

root.mainloop()

cv2.waitKey(0)
  
# closing all open windows
cv2.destroyAllWindows()
