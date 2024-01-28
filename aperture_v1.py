import cv2
import numpy as np


size = 1024
origin = [size//2, size//2]

canvas = np.zeros([size,size], dtype=np.uint8)

canvas = cv2.circle(canvas, origin, 256, 255, -1)
canvas = cv2.circle(canvas, origin, 64, 0, -1)
canvas = cv2.GaussianBlur(canvas, (25,25), 50)

print(np.max(canvas))

cv2.imshow("aperture", canvas)
cv2.waitKey(0)
cv2.destroyAllWindows()