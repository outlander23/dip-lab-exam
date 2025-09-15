# Add missing import
import matplotlib.pyplot as plt
# Chapter 1 – Introduction

# 1. Load and Display Images
# Assume 'grayscale_image.jpg' is a grayscale image and 'color_image.jpg' is a color image.
# If not grayscale, you can convert: gray_img = np.mean(plt.imread('color.jpg'), axis=-1)
gray_img = plt.imread('gray.jpeg')
color_img = plt.imread('color.webp')

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