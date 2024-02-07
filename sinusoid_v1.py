import cv2
import numpy as np

size = 512
freq = 32
step = .1
count = 0


while True:

    # Define the range and resolution of the grid
    x = np.linspace(-np.pi*freq*2, np.pi*freq*2, size)
    y = np.linspace(-np.pi*freq*2, np.pi*freq*2, size)

    # Create the meshgrid
    X, Y = np.meshgrid(x, y)

    # Calculate the radial distance from the origin
    R = np.sqrt(X**2 + Y**2)

    # Generate a radial sinusoid
    Z = np.sin(R + count/2*np.pi)  # Simple radial sinusoid

    # Display animation frame
    canvas = np.zeros([size,size], dtype=np.uint8)
    cv2.imshow("sinusoid", Z)
    if cv2.waitKey(2) == ord('q'):
        break

    count += step