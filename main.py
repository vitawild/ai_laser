import cv2
import numpy as np

image = cv2.imread('50.bmp', cv2.IMREAD_GRAYSCALE)
denoised = cv2.GaussianBlur(image, (5, 5), 0)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (65, 65))
bg = cv2.morphologyEx(denoised, cv2.MORPH_OPEN, kernel)

flat = cv2.subtract(denoised, bg)

c_flat = cv2.cvtColor(flat, cv2.COLOR_GRAY2BGR)
lab = cv2.cvtColor(c_flat, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_eq = clahe.apply(l)
lab_eq = cv2.merge((l_eq, a, b))
result = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

cv2.imwrite('result.jpg', result)
cv2.imshow('res', result)
cv2.waitKey(0)
