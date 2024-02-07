import cv2
import numpy as np
from numpy import inf

"""
Generate 3d fractal noise
"""
class Generate:
    # Generate 2d or 3d fractal
    def __init__(self, beta=4, seed=823, size=256, dimension=3, preview=False, save=False, method="ifft"):
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


"""
Aperture Mask - Animated Demo
"""

size = 256  # Window size
fractal = Generate(beta=4, size=size)
origin = [size//2, size//2]  # Center of canvas
corner = int(np.sqrt(((size//2)**2)*2))  # Distance between origin and corner

# Animation
loop = True
frame = 0
counter = 0
blur = 25

while loop:

    fractal_frame = np.array(fractal.pattern[frame,:,:]*255, dtype=np.float32)

    if counter < corner:

        canvas = np.zeros([size+1,size+1], dtype=np.float32)  # Black Canvas
        gray = np.ones([size+1,size+1], dtype=np.uint8)*128  # Gray Canvas
        mask = cv2.circle(canvas, origin, counter, 1, -1)
        mask = np.array(cv2.GaussianBlur(canvas, (blur,blur), blur))
        canvas = np.array(mask * fractal_frame, dtype=np.uint8)
        gray -= np.array(mask, dtype=np.uint8)
        canvas += gray

        cv2.imshow("aperture", canvas)
        cv2.waitKey(16)
        if cv2.waitKey(16) == ord('q'):
            break

    elif counter < corner*2:

        canvas = np.ones([size+1,size+1], dtype=np.float32)  # White Canvas
        gray = np.ones([size+1,size+1], dtype=np.uint8)*128  # Gray Canvas
        mask = cv2.circle(canvas, origin, counter-corner, 0, -1)
        mask = np.array(cv2.GaussianBlur(canvas, (blur,blur), blur))
        canvas = np.array(mask * fractal_frame, dtype=np.uint8)
        gray -= np.array(mask, dtype=np.uint8)
        canvas += gray



        cv2.imshow("aperture", canvas)
        cv2.waitKey(16)
        if cv2.waitKey(16) == ord('q'):
            break

    elif counter >= corner*2:
        counter = 0

    counter += 1
    frame += 1

    if frame >= size:
        frame = 0
