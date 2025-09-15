import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.fft import fft2, ifft2, fftshift, ifftshift

# Chapter 1 – Introduction

# 1. Load and Display Images
# Assume 'grayscale_image.jpg' is a grayscale image and 'color_image.jpg' is a color image.
# If not grayscale, you can convert: gray_img = np.mean(plt.imread('color.jpg'), axis=-1)
gray_img = plt.imread('grayscale_image.jpg')
color_img = plt.imread('color_image.jpg')

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Grayscale Image')
axs[1].imshow(color_img)
axs[1].set_title('Color Image')
plt.show()

# Differences: Grayscale has 1 channel (intensity 0-255), appears black/white. Color has 3 channels (RGB), full color.

# 2. Negative of an Image
# Assuming gray_img is uint8 0-255
negative_img = 255 - gray_img

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original Grayscale')
axs[1].imshow(negative_img, cmap='gray')
axs[1].set_title('Negative Image')
plt.show()

# Negative inverts intensities, making dark areas light and vice versa, affects contrast perception.

# 3. Image Subsampling
# Simple subsampling by slicing (for factors of 2,4,8)
sub2 = gray_img[::2, ::2]
sub4 = gray_img[::4, ::4]
sub8 = gray_img[::8, ::8]

fig, axs = plt.subplots(1, 4, figsize=(15, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(sub2, cmap='gray')
axs[1].set_title('Subsampled x2')
axs[2].imshow(sub4, cmap='gray')
axs[2].set_title('Subsampled x4')
axs[3].imshow(sub8, cmap='gray')
axs[3].set_title('Subsampled x8')
plt.show()

# Resolution decreases, details lost as subsampling increases (aliasing may occur).

# Chapter 2 – Digital Image Fundamentals

# 1. Gray-Level Quantization
# Function to quantize to n levels
def quantize(img, levels):
    return np.round(img / 255.0 * (levels - 1)) / (levels - 1) * 255

q2 = quantize(gray_img, 2)
q4 = quantize(gray_img, 4)
q8 = quantize(gray_img, 8)
q16 = quantize(gray_img, 16)
q256 = quantize(gray_img, 256)  # Original

fig, axs = plt.subplots(1, 5, figsize=(20, 5))
axs[0].imshow(q2, cmap='gray')
axs[0].set_title('2 levels')
axs[1].imshow(q4, cmap='gray')
axs[1].set_title('4 levels')
axs[2].imshow(q8, cmap='gray')
axs[2].set_title('8 levels')
axs[3].imshow(q16, cmap='gray')
axs[3].set_title('16 levels')
axs[4].imshow(q256, cmap='gray')
axs[4].set_title('256 levels')
plt.show()

# Fewer levels reduce quality, cause false contouring, loss of details.

# 2. Bit-Plane Slicing
bit_planes = []
for bit in range(8):
    plane = ((gray_img >> bit) & 1) * 255
    bit_planes.append(plane)

fig, axs = plt.subplots(2, 4, figsize=(15, 8))
for i in range(8):
    ax = axs[i // 4, i % 4]
    ax.imshow(bit_planes[7 - i], cmap='gray')  # MSB to LSB
    ax.set_title(f'Bit-plane {7 - i} (MSB to LSB)')
plt.show()

# Higher bit-planes (MSB) contain most structural info, lower (LSB) noise-like.

# 3. Pixel Neighborhood Analysis
# Choose a pixel, e.g., at (100, 100), assume image size >200
i, j = 100, 100
pixel_value = gray_img[i, j]

# 4-neighbors: top, bottom, left, right
four_neigh = []
if i > 0: four_neigh.append(gray_img[i-1, j])
if i < gray_img.shape[0]-1: four_neigh.append(gray_img[i+1, j])
if j > 0: four_neigh.append(gray_img[i, j-1])
if j < gray_img.shape[1]-1: four_neigh.append(gray_img[i, j+1])

# 8-neighbors: include diagonals
eight_neigh = four_neigh[:]
if i > 0 and j > 0: eight_neigh.append(gray_img[i-1, j-1])
if i > 0 and j < gray_img.shape[1]-1: eight_neigh.append(gray_img[i-1, j+1])
if i < gray_img.shape[0]-1 and j > 0: eight_neigh.append(gray_img[i+1, j-1])
if i < gray_img.shape[0]-1 and j < gray_img.shape[1]-1: eight_neigh.append(gray_img[i+1, j+1])

print(f"Pixel at ({i},{j}): {pixel_value}")
print("4-neighbors:", four_neigh)
print("8-neighbors:", eight_neigh)

# Neighborhoods used in local operations like filtering, edge detection.

# Chapter 3 – Intensity Transformations & Spatial Filtering

# 1. Histogram and Histogram Equalization
# Plot histogram
plt.hist(gray_img.ravel(), bins=256, range=(0, 256))
plt.title('Histogram')
plt.show()

# Manual histogram equalization
def hist_equalize(img):
    hist, bins = np.histogram(img.flatten(), 256, [0, 256])
    cdf = hist.cumsum()
    cdf_normalized = cdf * 255 / cdf[-1]
    img_eq = np.interp(img.flatten(), bins[:-1], cdf_normalized)
    return img_eq.reshape(img.shape).astype(np.uint8)

eq_img = hist_equalize(gray_img)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(eq_img, cmap='gray')
axs[1].set_title('Equalized')
plt.show()

plt.hist(eq_img.ravel(), bins=256, range=(0, 256))
plt.title('Equalized Histogram')
plt.show()

# Equalization spreads intensities, improves contrast/visibility.

# 2. Contrast Stretching
# Linear min-max stretch
min_val = np.min(gray_img)
max_val = np.max(gray_img)
stretched = ((gray_img - min_val) / (max_val - min_val) * 255).astype(np.uint8)

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original Low-Contrast')
axs[1].imshow(stretched, cmap='gray')
axs[1].set_title('Stretched')
plt.show()

# Stretching expands intensity range, enhances appearance.

# 3. Smoothing with Spatial Filters
# Averaging filter
smooth3 = ndimage.uniform_filter(gray_img, size=3)
smooth5 = ndimage.uniform_filter(gray_img, size=5)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(smooth3, cmap='gray')
axs[1].set_title('3x3 Average')
axs[2].imshow(smooth5, cmap='gray')
axs[2].set_title('5x5 Average')
plt.show()

# Larger kernel blurs more, reduces noise but loses details.

# 4. Sharpening with Laplacian Filter
laplacian = ndimage.laplace(gray_img)
sharpened = np.clip(gray_img + laplacian, 0, 255)  # Simple add back

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(sharpened, cmap='gray')
axs[1].set_title('Sharpened')
plt.show()

# Sharpening enhances edges, makes features sharper.

# Chapter 4 – Filtering in the Frequency Domain

# 1. Fourier Transform and Spectrum Visualization
f = fft2(gray_img)
fshift = fftshift(f)
magnitude_spectrum = np.log(np.abs(fshift) + 1)

plt.imshow(magnitude_spectrum, cmap='gray')
plt.title('Magnitude Spectrum (Log Scaled)')
plt.show()

# Low freq center: smooth areas, high freq: edges/details.

# 2. Low-Pass Filtering in Frequency Domain
# Ideal Low-Pass Filter
rows, cols = gray_img.shape
crow, ccol = rows//2, cols//2
radius = 30  # Cutoff frequency

mask = np.zeros((rows, cols), np.uint8)
x, y = np.ogrid[:rows, :cols]
mask_area = (x - crow)**2 + (y - ccol)**2 <= radius**2
mask[mask_area] = 1

fshift_lpf = fshift * mask
f_ishift = ifftshift(fshift_lpf)
img_lpf = np.real(ifft2(f_ishift))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(img_lpf, cmap='gray')
axs[1].set_title('Low-Pass Filtered')
plt.show()

# Removes high freq, blurs image.

# 3. High-Pass Filtering in Frequency Domain
# Ideal High-Pass Filter
mask_hpf = 1 - mask  # Invert LPF

fshift_hpf = fshift * mask_hpf
f_ishift_hpf = ifftshift(fshift_hpf)
img_hpf = np.real(ifft2(f_ishift_hpf))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(gray_img, cmap='gray')
axs[0].set_title('Original')
axs[1].imshow(img_hpf, cmap='gray')
axs[1].set_title('High-Pass Filtered')
plt.show()

# Highlights high freq, enhances edges.

# 4. Notch Filtering for Periodic Noise Removal
# Add synthetic periodic noise (sinusoidal)
xx, yy = np.meshgrid(np.arange(cols), np.arange(rows))
noise = 50 * np.sin(2 * np.pi * xx / 30 + 2 * np.pi * yy / 30)  # Example freq
noisy_img = np.clip(gray_img + noise, 0, 255)

# Fourier of noisy
f_noisy = fft2(noisy_img)
fshift_noisy = fftshift(f_noisy)

# Notch filter: set specific freq to zero (assume known locations, e.g., from spectrum)
# For example, notch at certain points
notch_mask = np.ones_like(fshift_noisy)
# Assume noise spikes at (crow + 10, ccol + 10) and symmetric
notch_radius = 5
for dx, dy in [(10,10), (-10,-10), (10,-10), (-10,10)]:  # Example positions
    notch_area = (x - (crow + dy))**2 + (y - (ccol + dx))**2 <= notch_radius**2
    notch_mask[notch_area] = 0

fshift_notch = fshift_noisy * notch_mask
f_ishift_notch = ifftshift(fshift_notch)
img_notch = np.real(ifft2(f_ishift_notch))

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
axs[0].imshow(noisy_img, cmap='gray')
axs[0].set_title('Noisy Image')
axs[1].imshow(img_notch, cmap='gray')
axs[1].set_title('Notch Filtered')
plt.show()

# Notch removes specific freq components causing periodic noise.