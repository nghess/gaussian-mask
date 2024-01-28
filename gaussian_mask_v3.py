import numpy as np
from numpy import inf
import cv2

"""
Generate 3d Fractal
"""

class Generate:
    # Generate 2d or 3d fractal
    def __init__(self, beta=4, seed=117, size=256, dimension=3, preview=False, save=False, method="ifft"):
        # Set Seed
        np.random.seed(seed)
        # Set Size
        size = size+1
        # Set properties
        self.beta = beta
        self.seed = seed
        self.size = size
        self.dimension = dimension

        #Alert related
        assert self.dimension == 2 or self.dimension == 3, "Dimension must be either 2 or 3"
        np.seterr(divide='ignore')

        if dimension == 2 and method == "ifft":
            # Build power spectrum
            f = [x/size for x in range(0, int(size/2)+1)] + [x/size for x in range(-int(size/2), 0)]
            u = np.reshape(f, (size, 1))
            v = np.reshape(f, (1, size))
            powerspectrum = (u**2 + v**2)**(-beta/2)
            powerspectrum[powerspectrum == inf] = powerspectrum[0,1]
            # Noise and ifft
            phases = np.random.normal(0, 255, size=[size, size])
            pattern = np.fft.ifftn(powerspectrum**0.5 * (np.cos(2*np.pi*phases)+1j*np.sin(2*np.pi*phases)))
            # Normalize result
            pattern = np.real(pattern)
            self.pattern = (pattern-np.amin(pattern))/np.amax(pattern-np.amin(pattern))

        if dimension == 3 and method == "ifft":
            # Pre-compute constants and arrays
            f = np.fft.fftfreq(size)  # Use fftfreq to generate frequency bins directly
            f = f[:size + 1]  # Take only the non-negative frequencies (assuming size is even)
            
            # Broadcasting to generate u, v, w without explicit reshaping
            f2 = f**2
            powerspectrum = (f2[:, None, None] + f2[None, :, None] + f2[None, None, :])**(-beta / 2)
            
            # Handle infinity
            powerspectrum[np.isinf(powerspectrum)] = np.nan
            np.nan_to_num(powerspectrum, copy=False, nan=powerspectrum[0, 1, 0])

            # Ensure the powerspectrum is in single precision (complex64)
            powerspectrum_single = np.sqrt(powerspectrum).astype(np.complex64)

            # Combine powerspectrum with phases and convert to single precision
            phases_single = np.random.normal(0, 255, (size, size, size)).astype(np.float32)
            complex_phase = np.cos(2 * np.pi * phases_single) + 1j * np.sin(2 * np.pi * phases_single)
            complex_phase = complex_phase.astype(np.complex64)

            pattern = np.fft.ifftn(powerspectrum_single * complex_phase)
            
            # Normalize result
            pattern = np.real(pattern)
            pattern -= pattern.min()
            pattern /= pattern.max()

            self.pattern = pattern

    def previewAnim(self, reps=3, mode='gs'):
        if reps == 1:
            reps = 2
        for i in range(reps-1):
            for k in range(self.size):
                cv2.imshow('Fractal Preview', self.pattern[k, :, :])
                cv2.waitKey(16)

def generate_mask(size, radius=64, sigma=20.0):
    size = size + 1
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

    return ((g-np.amin(g))/np.amax(g-np.amin(g))).reshape(1, size, size)


"""
Generate Stimulus
"""

size = 512

fractal = Generate(beta=5, size=size)
mask = generate_mask(size=size)

fractal.pattern = fractal.pattern * mask

fractal.previewAnim()