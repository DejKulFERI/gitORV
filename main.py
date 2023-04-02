import cv2

img = cv2.imread('images/Slika_1.jpg')
img = cv2.resize(img, (600, 500))

cv2.imshow('Image', img)

cv2.waitKey(0)
cv2.destroyAllWindows()
