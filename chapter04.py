import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os

# Chapter 4 – Filtering in the Frequency Domain

def load_grayscale(path):
    img = imread(path)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3:
        img = (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(np.uint8)
    return img


def spectrum(img):
    # img: 2D uint8
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    return f, fshift, magnitude


def show_log_spectrum(magnitude, ax=None, title='Magnitude Spectrum (log)'):
    mag_log = np.log1p(magnitude)
    if ax is None:
        plt.imshow(mag_log, cmap='gray')
        plt.title(title)
        plt.axis('off')
    else:
        ax.imshow(mag_log, cmap='gray')
        ax.set_title(title)
        ax.axis('off')


def ideal_lowpass(shape, D0):
    M, N = shape
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
    V, U = np.meshgrid(v, u)
    D = np.sqrt(U**2 + V**2)
    H = np.zeros_like(D)
    H[D <= D0] = 1
    return H


def ideal_highpass(shape, D0):
    return 1 - ideal_lowpass(shape, D0)


def apply_filter_freqdomain(img, H):
    F = np.fft.fft2(img)
    Fshift = np.fft.fftshift(F)
    Gshift = Fshift * H
    G = np.fft.ifftshift(Gshift)
    img_back = np.fft.ifft2(G)
    img_back = np.real(img_back)
    img_back = np.clip(img_back, 0, 255).astype(np.uint8)
    return img_back


def add_sinusoidal_noise(img, freq=(10,0), amplitude=30):
    # freq: (u_freq, v_freq) in cycles per image dimension
    M, N = img.shape
    u_freq, v_freq = freq
    x = np.arange(M)[:,None]
    y = np.arange(N)[None,:]
    pattern = np.sin(2*np.pi*(u_freq * x / M + v_freq * y / N))
    noisy = img.astype(np.float32) + amplitude * pattern
    noisy = np.clip(noisy, 0, 255).astype(np.uint8)
    return noisy


def notch_filter(shape, centers, radius):
    # centers: list of (u_center, v_center) in shifted coordinate system (center at 0)
    M, N = shape
    u = np.arange(-M//2, M//2)
    v = np.arange(-N//2, N//2)
    V, U = np.meshgrid(v, u)
    H = np.ones((M, N), dtype=np.float32)
    for (uc, vc) in centers:
        D = np.sqrt((U - uc)**2 + (V - vc)**2)
        H[D <= radius] = 0
    return H


if __name__ == '__main__':
    candidates = ['gray.jpeg', 'grayscale_image.jpg', 'gray.png', 'color.webp', 'color_image.jpg']
    img_path = None
    for c in candidates:
        if os.path.exists(c):
            img_path = c
            break
    if img_path is None:
        raise FileNotFoundError('No image found. Put a grayscale or color image named one of: ' + ','.join(candidates))

    img = load_grayscale(img_path)
    M, N = img.shape

    # 1. Fourier Transform and Spectrum Visualization
    F, Fshift, mag = spectrum(img)

    fig, axs = plt.subplots(1,2,figsize=(12,5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original Image')
    axs[0].axis('off')
    show_log_spectrum(mag, ax=axs[1], title='Log Magnitude Spectrum')
    plt.suptitle('FFT and Magnitude Spectrum')
    plt.show()

    print('\nSpectrum discussion:')
    print('Low-frequency components (near center after shifting) correspond to smooth variations and overall brightness. High-frequency components represent edges and fine detail.')

    # 2. Ideal Low-Pass Filter (ILPF)
    D0 = min(M, N) // 8
    H_lp = ideal_lowpass((M, N), D0)
    img_lp = apply_filter_freqdomain(img, H_lp)

    fig, axs = plt.subplots(1,2,figsize=(12,5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(img_lp, cmap='gray')
    axs[1].set_title(f'ILPF D0={D0}')
    axs[1].axis('off')
    plt.suptitle('Ideal Low-Pass Filtering')
    plt.show()

    print('\nLow-pass discussion:')
    print('Low-pass filtering removes high-frequency components, which blurs edges and fine details while preserving smooth regions.')

    # 3. Ideal High-Pass Filter (IHPF)
    D0_hp = min(M, N) // 16
    H_hp = ideal_highpass((M, N), D0_hp)
    img_hp = apply_filter_freqdomain(img, H_hp)

    fig, axs = plt.subplots(1,2,figsize=(12,5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(img_hp, cmap='gray')
    axs[1].set_title(f'IHPF D0={D0_hp}')
    axs[1].axis('off')
    plt.suptitle('Ideal High-Pass Filtering')
    plt.show()

    print('\nHigh-pass discussion:')
    print('High-pass filtering highlights edges and fine details by removing the low-frequency content (smooth variations).')

    # 4. Notch Filtering for Periodic Noise Removal
    # Add synthetic sinusoidal noise
    noisy = add_sinusoidal_noise(img, freq=(6, 0), amplitude=40)

    # Compute spectrum of noisy image to locate peaks
    _, Fshift_noisy, mag_noisy = spectrum(noisy)

    # We know noise introduces symmetric peaks at +/- frequency along u axis. Estimate centers.
    # For robustness, compute coordinates of top peaks excluding center.
    mag_copy = mag_noisy.copy()
    center = (M//2, N//2)
    mag_copy[center[0]-5:center[0]+6, center[1]-5:center[1]+6] = 0
    # find peaks
    idx = np.unravel_index(np.argsort(mag_copy.ravel())[-8:], mag_copy.shape)
    centers = list(zip(idx[0] - M//2, idx[1] - N//2))
    # Keep only pairs with magnitude above a small threshold
    # We'll pick the two strongest symmetric peaks
    # Sort centers by magnitude
    peak_coords = list(zip(idx[0], idx[1]))
    peak_coords = sorted(peak_coords, key=lambda rc: mag_copy[rc], reverse=True)
    selected = peak_coords[:2]
    centers_shifted = [(r - M//2, c - N//2) for (r, c) in selected]
    # Add symmetric counterparts
    centers_full = centers_shifted + [(-r, -c) for (r, c) in centers_shifted]

    radius = 5
    H_notch = notch_filter((M, N), centers_full, radius)
    noisy_filtered = apply_filter_freqdomain(noisy, H_notch)

    fig, axs = plt.subplots(2,2,figsize=(12,10))
    axs[0,0].imshow(noisy, cmap='gray')
    axs[0,0].set_title('Noisy Image (sinusoidal)')
    axs[0,0].axis('off')
    show_log_spectrum(mag_noisy, ax=axs[0,1], title='Noisy Spectrum (log)')

    axs[1,0].imshow(noisy_filtered, cmap='gray')
    axs[1,0].set_title('After Notch Filtering')
    axs[1,0].axis('off')
    # show filter mask magnitude (visualize H_notch)
    axs[1,1].imshow(H_notch, cmap='gray')
    axs[1,1].set_title('Notch Filter Mask')
    axs[1,1].axis('off')
    plt.suptitle('Notch Filtering for Periodic Noise Removal')
    plt.show()

    print('\nNotch filtering discussion:')
    print('Notch filters remove energy at specific frequency locations (peaks) introduced by periodic noise. By zeroing those bins (and symmetric counterparts), the periodic pattern is suppressed.')
