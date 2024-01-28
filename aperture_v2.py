import cv2
import numpy as np


size = 1024
origin = [size//2, size//2]  # Center of canvas
corner = int(np.sqrt(((size//2)**2)*2))  # Distance between origin and corner





for ii in range(corner):

    canvas = np.zeros([size,size], dtype=np.uint8)
    canvas = cv2.circle(canvas, origin, ii, 255, -1)
    canvas = cv2.GaussianBlur(canvas, (25,25), 50)

    cv2.imshow("aperture", canvas)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break

for ii in range(corner):

    canvas = cv2.circle(canvas, origin, ii, 0, -1)
    canvas = cv2.GaussianBlur(canvas, (25,25), 50)

    cv2.imshow("aperture", canvas)
    cv2.waitKey(1)
    if cv2.waitKey(1) == ord('q'):
        break