import numpy as np
import matplotlib.pyplot as plt

# Chapter 2 – Digital Image Fundamentals

# Utility: load a grayscale image (fall back to converting a color if needed)
from matplotlib.image import imread


def load_grayscale(path):
    img = imread(path)
    # If image is float in [0,1], convert to uint8
    if img.dtype == np.float32 or img.dtype == np.float64:
        img = (img * 255).astype(np.uint8)
    if img.ndim == 3:
        # convert to grayscale by luminosity
        img = (0.2989 * img[..., 0] + 0.5870 * img[..., 1] + 0.1140 * img[..., 2]).astype(np.uint8)
    return img


# 1. Gray-Level Quantization

def quantize(img, levels):
    # img: uint8
    assert img.dtype == np.uint8
    # Map 0-255 to 0-(levels-1), then back to 0-255
    q = (img.astype(np.float32) / 255.0) * (levels - 1)
    q = np.round(q)
    q = (q / (levels - 1)) * 255.0
    return q.astype(np.uint8)


# 2. Bit-Plane Slicing

def bit_planes(img):
    # return list of 8 images (0=LSB to 7=MSB)
    planes = []
    for b in range(8):
        plane = ((img >> b) & 1) * 255
        planes.append(plane.astype(np.uint8))
    return planes[::-1]  # return MSB->LSB


# 3. Pixel Neighborhood Analysis

def pixel_neighbors(img, r, c):
    h, w = img.shape
    four = []
    eight = []
    # 4-neighbors: top, bottom, left, right
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        rr, cc = r + dr, c + dc
        if 0 <= rr < h and 0 <= cc < w:
            four.append(((rr, cc), int(img[rr, cc])))
    # 8-neighbors: include diagonals
    for dr in [-1, 0, 1]:
        for dc in [-1, 0, 1]:
            if dr == 0 and dc == 0:
                continue
            rr, cc = r + dr, c + dc
            if 0 <= rr < h and 0 <= cc < w:
                eight.append(((rr, cc), int(img[rr, cc])))
    return four, eight


if __name__ == '__main__':
    # Try common grayscale filename; fall back to converting color.webp
    import os
    candidates = ['gray.jpeg', 'grayscale_image.jpg', 'gray.png', 'color.webp', 'color_image.jpg']
    img_path = None
    for c in candidates:
        if os.path.exists(c):
            img_path = c
            break
    if img_path is None:
        raise FileNotFoundError('No image found. Put a grayscale or color image in the working directory named one of: ' + ','.join(candidates))

    gray = load_grayscale(img_path)

    # 1. Quantization
    levels_list = [2, 4, 8, 16, 256]
    quantized = [quantize(gray, L) for L in levels_list]

    fig, axs = plt.subplots(1, len(levels_list) + 1, figsize=(15, 4))
    axs[0].imshow(gray, cmap='gray')
    axs[0].set_title('Original (256 levels)')
    axs[0].axis('off')
    for i, L in enumerate(levels_list):
        axs[i + 1].imshow(quantized[i], cmap='gray', vmin=0, vmax=255)
        axs[i + 1].set_title(f'{L} levels')
        axs[i + 1].axis('off')
    plt.suptitle('Gray-Level Quantization')
    plt.show()

    print('\nQuantization discussion:')
    print('Reducing number of gray levels causes banding and loss of subtle gradients; fine details may vanish as levels decrease.')

    # 2. Bit-plane slicing
    planes = bit_planes(gray)
    fig, axs = plt.subplots(2, 4, figsize=(12, 6))
    for i, ax in enumerate(axs.ravel()):
        ax.imshow(planes[i], cmap='gray', vmin=0, vmax=255)
        ax.set_title(f'Bit {7-i} (MSB->LSB)')
        ax.axis('off')
    plt.suptitle('Bit-Plane Slicing (MSB -> LSB)')
    plt.show()

    print('\nBit-plane discussion:')
    print('Higher-order bit-planes (MSB) contain most of the structural information and low-frequency content. Lower bit-planes add fine details and noise.')

    # 3. Pixel neighborhood analysis
    r, c = gray.shape[0] // 2, gray.shape[1] // 2
    four, eight = pixel_neighbors(gray, r, c)
    print('\nPixel Neighborhood Analysis:')
    print(f'Chosen pixel: ({r}, {c}) value={int(gray[r, c])}')
    print('4-neighbors (pos, value):', four)
    print('8-neighbors (pos, value):', eight)
    print('\nNeighborhoods are used in local filtering, edge detection, and morphological operations; the 4-neighbors capture orthogonal adjacency while 8-neighbors include diagonals.')
