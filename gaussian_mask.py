import numpy as np
import cv2


def gaussian_2d(x, y, mu_x=0, mu_y=0, sigma_x=5, sigma_y=5):

    return 1 / (2 * np.pi * sigma_x * sigma_y) * np.exp(-((x - mu_x)**2 / (2 * sigma_x**2) + (y - mu_y)**2 / (2 * sigma_y**2)))


m = 1024
n = 1024

canvas = np.zeros([m,n], dtype=np.uint8)


gaussian = canvas

for x in range(m):
    for y in range(n):
        gaussian[x,y] = gaussian_2d(x,y, mu_x=512, mu_y=512, sigma_x=10, sigma_y=10)*255


print(np.max(gaussian))

cv2.imshow("gaussian", gaussian)
cv2.waitKey(0)
cv2.destroyAllWindows()