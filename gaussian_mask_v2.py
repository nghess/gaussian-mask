import numpy as np
import cv2

def generate_2d_gaussian(size, radius=800, sigma=50.0):
    size = size+1
    x = np.linspace(-size / 2, size / 2, size)
    y = np.linspace(-size / 2, size / 2, size)
    x, y = np.meshgrid(x, y)

    # Convert Cartesian coordinates to polar coordinates
    Theta = np.arctan2(y, x)
    RadialDistance = np.sqrt(x**2 + y**2)

    # Calculate the distance of each point to the nearest point on the central circle
    DistanceToCircle = np.abs(RadialDistance - radius)

    # 2D Gaussian function using the distance to the central circle
    g = np.exp(-(DistanceToCircle**2) / (2 * sigma**2))

    # d = np.sqrt(x*x + y*y)
    # g = np.exp(-( (d)**2 / ( 2.0 * sigma**2 ) ) )
    return (g-np.amin(g))/np.amax(g-np.amin(g))

def generate_fractal(beta, sdim):
    
    # Set Seed
    np.random.seed(117)
    # Turn of div by zero warning
    np.seterr(divide='ignore')

    # Build power spectrum
    sdim += 1
    f = [x/sdim for x in range(0,int(sdim/2)+1)] + [x/sdim for x in range(-int(sdim/2),0)]
    u = np.reshape(f, (sdim, 1))
    v = np.reshape(f, (1, sdim))
    powerspectrum = (u**2 + v**2)**(-beta/2)

    # Patch any infinities
    powerspectrum[powerspectrum == np.inf] = powerspectrum[0,1]

    # Noise and ifft
    phases = np.random.normal(127.5, 64, size=[sdim, sdim])
    complexpattern = np.fft.ifft2(powerspectrum**0.5 * (np.cos(2*np.pi*phases)+1j*np.sin(2*np.pi*phases)))

    # Get real part, then normalize result between 0 and 1
    realpattern = np.real(complexpattern)
    return (realpattern-np.amin(realpattern))/np.amax(realpattern-np.amin(realpattern))


size = 1024
sigma = 50
radius = 300
beta = 3.5

mask = generate_2d_gaussian(size, radius, sigma)
print(np.max(mask))

stimulus = (generate_fractal(beta, size) * mask)

# Normalize the Gaussian array to have values between 0 and 255
gaussian_normalized = cv2.normalize(stimulus, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1)

# Display the image
cv2.imshow("2D Gaussian distribution", gaussian_normalized)
cv2.waitKey(0)
cv2.destroyAllWindows()
