import cv2
import numpy as np

image = cv2.imread('01.bmp')
if image is None:
    raise ValueError("не удалось загрузить изображение")

h, w = image.shape[:2]
camera_matrix = np.array([[w, 0, w/2], [0, h, h/2], [0, 0, 1]], dtype=np.float32)
dist_coeffs = np.zeros(4)
undistorted = cv2.undistort(image, camera_matrix, dist_coeffs)

lab = cv2.cvtColor(undistorted, cv2.COLOR_BGR2LAB)
l, a, b = cv2.split(lab)
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
l_eq = clahe.apply(l)
lab_eq = cv2.merge((l_eq, a, b))
first_res = cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)

gray = cv2.cvtColor(first_res, cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray, (9, 9), 2)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, dp=1, minDist=50,
                         param1=50, param2=30, minRadius=10, maxRadius=30)

if circles is None or len(circles[0]) != 4:
    raise ValueError("не удалось обнаружить 4 реперные точки")

circles = np.uint16(np.around(circles[0]))

circles = sorted(circles, key=lambda x: (x[1], x[0]))
img_pts = np.array([(x[0], x[1]) for x in circles], dtype=np.float32)


obj_pts = np.array([[5, 5], [65, 5], [65, 65], [5, 65]], dtype=np.float32)

H, _ = cv2.findHomography(img_pts, obj_pts)

final_warped = cv2.warpPerspective(first_res, H, (70, 70))

gray_warped = cv2.cvtColor(final_warped, cv2.COLOR_BGR2GRAY)
_, thresh = cv2.threshold(gray_warped, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


max_contour = max(contours, key=cv2.contourArea)

rect = cv2.minAreaRect(max_contour)
box = cv2.boxPoints(rect)
box = np.int0(box)

center, size, angle = rect
if size[0] < size[1]:
    angle += 90

marked = first_res.copy()
for i, (x, y, r) in enumerate(circles):
    cv2.circle(marked, (x, y), r, (0, 255, 0), 2)
    cv2.putText(marked, str(i+1), (x-10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

cv2.drawContours(final_warped, [box], 0, (0, 0, 255), 2)

print(f"координаты центра: X={center[0]:.1f} мм, Y={center[1]:.1f} мм")
print(f"угол поворота: {angle:.1f}°")

cv2.imwrite('first_res.jpg', marked)
cv2.imwrite('aligned_result.jpg', final_warped)

cv2.imshow('first res with markers', marked)
cv2.imshow('result with contour', final_warped)
cv2.waitKey(0)
cv2.destroyAllWindows()
