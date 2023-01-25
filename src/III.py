import numpy as np
from scipy.fftpack import dct, idct
from PIL import Image
import zlib

# Load image
im = Image.open(
    r"C:\Users\Usuario\OneDrive\Bureau\Project VCF\env/image.png").convert('L')

# Convert the image to a numpy array and reshape it to a 2D array of size (height, width)
im_array = np.array(im)
im_array = im_array.reshape((im.height, im.width))

# Perform DCT
dct_coefficients = dct(dct(im_array, axis=0), axis=1)
# Divide the image into 8x8 blocks
dct_coefficients_8x8 = dct_coefficients.reshape(-1, 8, 8)


# Quantization
Q = np.array([[16, 11, 10, 16, 24, 40, 51, 61],
              [12, 12, 14, 19, 26, 58, 60, 55],
              [14, 13, 16, 24, 40, 57, 69, 56],
              [14, 17, 22, 29, 51, 87, 80, 62],
              [18, 22, 37, 56, 68, 109, 103, 77],
              [24, 35, 55, 64, 81, 104, 113, 92],
              [49, 64, 78, 87, 103, 121, 120, 101],
              [72, 92, 95, 98, 112, 100, 103, 99]])
print(dct_coefficients_8x8.shape)
print(dct_coefficients_8x8.dtype)
print(Q.shape)
print(Q.dtype)

quantized_coefficients = np.floor(dct_coefficients_8x8 / Q).astype(Q.dtype)


# Encode the quantized coefficients using zlib
compressed_data = zlib.compress(quantized_coefficients)

# Save the compressed data and quantization matrix to file
with open("compressed_image.bin", "wb") as f:
    f.write(compressed_data)
    f.write(Q)
# deocede:Read compressed data and quantization matrix from file
with open("compressed_image.bin", "rb") as f:
    compressed_data = f.read()
    quantization_matrix = f.read()

# Decompress the data using zlib
quantized_coefficients = zlib.decompress(compressed_data)

# Dequantization
dct_coefficients = dct_coefficients_8x8 * Q[np.newaxis, :, :]


# Perform IDCT
im_array_rec = idct(idct(dct_coefficients, axis=0), axis=1)
original_shape = im.size
# Create image from the array and display it
im_array_rec = im_array_rec.reshape(original_shape)
print(im_array_rec)
im_rec = Image.fromarray(im_array_rec.astype(np.uint8))
im_rec.show()
