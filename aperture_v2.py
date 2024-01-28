import cv2
import numpy as np

"""
Aperture Mask - Animated Demo
"""

size = 1024  # Window size
origin = [size//2, size//2]  # Center of canvas
corner = int(np.sqrt(((size//2)**2)*2))  # Distance between origin and corner
canvas = np.zeros([size,size], dtype=np.uint8)  # Array to draw on

# Animation
loop = True
counter = 0

while loop:

    if counter < corner:

        canvas = cv2.circle(canvas, origin, counter, 255, -1)
        canvas = cv2.GaussianBlur(canvas, (25,25), 50)

        cv2.imshow("aperture", canvas)
        cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            break

    elif counter < corner*2:

        canvas = cv2.circle(canvas, origin, counter-corner, 0, -1)
        canvas = cv2.GaussianBlur(canvas, (25,25), 50)

        cv2.imshow("aperture", canvas)
        cv2.waitKey(1)
        if cv2.waitKey(1) == ord('q'):
            break

    elif counter >= corner*2:
        counter = 0

    counter += 1

