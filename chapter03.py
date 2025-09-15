import numpy as np
import matplotlib.pyplot as plt
from matplotlib.image import imread
import os
from scipy import ndimage

# Chapter 3 – Intensity Transformations & Spatial Filtering

def load_grayscale(path):
    img = imread(path)
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3:
        img = (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(np.uint8)
    return img


# 1. Histogram and Histogram Equalization
def histogram(img):
    hist, bins = np.histogram(img.flatten(), bins=256, range=[0,255])
    return hist, bins


def hist_equalize(img):
    # img: uint8
    hist, bins = histogram(img)
    cdf = hist.cumsum()
    cdf_normalized = (cdf - cdf.min()) * 255 / (cdf.max() - cdf.min())
    cdf_normalized = cdf_normalized.astype(np.uint8)
    img2 = cdf_normalized[img]
    return img2


# 2. Contrast Stretching
def contrast_stretch(img, in_min=None, in_max=None):
    if in_min is None:
        in_min = img.min()
    if in_max is None:
        in_max = img.max()
    stretched = (img.astype(np.float32) - in_min) * (255.0 / (in_max - in_min))
    stretched = np.clip(stretched, 0, 255)
    return stretched.astype(np.uint8)


# 3. Smoothing with averaging filters
def average_filter(img, k):
    kernel = np.ones((k, k), dtype=np.float32) / (k * k)
    return ndimage.convolve(img, kernel, mode='reflect').astype(np.uint8)


# 4. Sharpening with Laplacian
def laplacian_sharpen(img):
    # standard 3x3 Laplacian kernel
    kernel = np.array([[0, 1, 0], [1, -4, 1], [0, 1, 0]])
    lap = ndimage.convolve(img.astype(np.int16), kernel, mode='reflect')
    sharp = img.astype(np.int16) - lap
    sharp = np.clip(sharp, 0, 255).astype(np.uint8)
    return sharp


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

    # 1. Histogram & equalization
    hist_orig, bins = histogram(img)
    img_eq = hist_equalize(img)
    hist_eq, _ = histogram(img_eq)

    fig, axs = plt.subplots(2, 2, figsize=(10, 8))
    axs[0,0].imshow(img, cmap='gray')
    axs[0,0].set_title('Original Image')
    axs[0,0].axis('off')
    axs[0,1].bar(range(256), hist_orig, color='black')
    axs[0,1].set_title('Original Histogram')

    axs[1,0].imshow(img_eq, cmap='gray')
    axs[1,0].set_title('Equalized Image')
    axs[1,0].axis('off')
    axs[1,1].bar(range(256), hist_eq, color='black')
    axs[1,1].set_title('Equalized Histogram')
    plt.suptitle('Histogram and Histogram Equalization')
    plt.show()

    print('\nHistogram Equalization discussion:')
    print('Equalization spreads out the intensity distribution, improving contrast especially in images with clustered intensities.')

    # 2. Contrast stretching (assume low contrast if dynamic range small)
    in_min, in_max = np.percentile(img, 2), np.percentile(img, 98)
    stretched = contrast_stretch(img, in_min=in_min, in_max=in_max)

    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original (low contrast)')
    axs[0].axis('off')
    axs[1].imshow(stretched, cmap='gray')
    axs[1].set_title('Contrast Stretched')
    axs[1].axis('off')
    plt.suptitle('Contrast Stretching')
    plt.show()

    print('\nContrast Stretching discussion:')
    print('Linear stretching expands the intensity range, increasing visibility of features but can amplify noise.')

    # 3. Smoothing with averaging filters
    avg3 = average_filter(img, 3)
    avg5 = average_filter(img, 5)

    fig, axs = plt.subplots(1,3,figsize=(12,5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(avg3, cmap='gray')
    axs[1].set_title('Averaging 3x3')
    axs[1].axis('off')
    axs[2].imshow(avg5, cmap='gray')
    axs[2].set_title('Averaging 5x5')
    axs[2].axis('off')
    plt.suptitle('Smoothing with Averaging Filters')
    plt.show()

    print('\nSmoothing discussion:')
    print('Larger kernels produce stronger blurring, reducing noise but also smoothing out fine details.')

    # 4. Laplacian sharpening
    sharp = laplacian_sharpen(img)
    fig, axs = plt.subplots(1,2,figsize=(10,5))
    axs[0].imshow(img, cmap='gray')
    axs[0].set_title('Original')
    axs[0].axis('off')
    axs[1].imshow(sharp, cmap='gray')
    axs[1].set_title('Laplacian Sharpened')
    axs[1].axis('off')
    plt.suptitle('Laplacian Sharpening')
    plt.show()

    print('\nSharpening discussion:')
    print('Laplacian sharpening highlights edges and high-frequency components, making features crisper but may amplify noise.')
