import numpy as np
import cv2
from scipy.fftpack import dct, idct


# Load image sequence
images = []
for i in range(1, 11):
    img = cv2.imread('image{}.jpg'.format(i))
    images.append(img)

# oyther methode:Input image sequence
#images = [np.random.rand(256, 256) for i in range(10)]

# Quantization table
quant_table = np.random.rand(8, 8)

# Huffman coding dictionary
huffman_dict = {i: i for i in range(256)}
encoded_data = []
for img in images:
    # Divide image into 8x8 blocks
    blocks = np.array([img[i:i+8, j:j+8] for i in range(0, img.shape[0], 8)
                      for j in range(0, img.shape[1], 8)])

    # Perform DCT on each block
    dct_blocks = np.array([dct(block, norm='ortho') for block in blocks])

    # Quantize the DCT coefficients
    quant_blocks = np.round(dct_blocks / quant_table)

    # Encode the quantized DCT coefficients using Huffman coding
    encoded_blocks = [huffman_dict[coeff]
                      for block in quant_blocks for coeff in block.flatten()]
    encoded_data.extend(encoded_blocks)

# Send encoded data over a channel

decoded_images = []
for i in range(0, len(encoded_data), 64*64):
    # Decode the data using the Huffman decoding algorithm
    decoded_blocks = [huffman_dict[coeff] for coeff in encoded_data[i:i+64*64]]

    # De-quantize the DCT coefficients
    dct_blocks = decoded_blocks * quant_table

    # Perform IDCT on each block
    blocks = np.array([idct(block, norm='ortho')
                      for block in dct_blocks]).reshape(-1, 8, 8)

    # Assemble the blocks back into the original image
    img = np.vstack([np.hstack(blocks[i:i+64])
                    for i in range(0, len(blocks), 64)])
    decoded_images.append(img)
