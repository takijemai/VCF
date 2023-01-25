import numpy as np
import cv2
from scipy.fftpack import dct, idct


# Load image sequence
images = []
for i in range(1, 11):
    img = cv2.imread(
        r'C:\Users\Usuario\OneDrive\Bureau\Project VCF\env/image{}.jpg'.format(i))
    if img.shape[0] % 8 == 0 and img.shape[1] % 8 == 0:
        images.append(img)
    else:
        print("Skipping image with shape {}".format(img.shape))


# oyther methode:Input image sequence
#images = [np.random.rand(256, 256) for i in range(10)]

# Quantization table
quant_table = np.random.rand(8, 8)

# Huffman coding dictionary
huffman_dict = {i: i for i in range(256)}
encoded_data = []
for img in images:
    if img.shape[0] % 8 == 0 and img.shape[1] % 8 == 0:
        for channel in range(3):
            # Divide image into 8x8 blocks
            blocks = np.array([img[i:i+8, j:j+8, channel]
                              for i in range(0, img.shape[0], 8) for j in range(0, img.shape[1], 8)])

    # Perform DCT on each block
    dct_blocks = np.array([dct(block, norm='ortho') for block in blocks])

    # Quantize the DCT coefficients
    quant_blocks = np.round(dct_blocks / quant_table)
    # Make sure the coefficients are non-negative before looking them up in the Huffman dictionary
    quant_blocks = np.abs(quant_blocks)
    # Make sure the coefficients are within the range of the Huffman dictionary
    quant_blocks[quant_blocks > 255] = 255
    # Encode the quantized DCT coefficients using Huffman coding
    encoded_blocks = [huffman_dict[coeff]
                      for block in quant_blocks for coeff in block.flatten()]
    encoded_data.extend(encoded_blocks)

# Send encoded data over a channel

decoded_images = []
for i in range(0, len(encoded_data), 64*64):
    # Decode the data using the Huffman decoding algorithm
    decoded_blocks = [huffman_dict[coeff] for coeff in encoded_data[i:i+64*64]]
    # reshape the decoded_blocks array
    decoded_blocks = np.array(decoded_blocks).reshape(-1, 8, 8)

    # De-quantize the DCT coefficients
    dct_blocks = decoded_blocks * quant_table

    # Perform IDCT on each block
    blocks = np.array([idct(block, norm='ortho')
                      for block in dct_blocks]).reshape(-1, 8, 8)

    # Assemble the blocks back into the original image
    img = np.vstack([np.hstack(blocks[i:i+64])
                    for i in range(0, len(blocks), 64)])
    decoded_images.append(img)

    # Display original image
    cv2.imshow("Original Image", images[0])

    # Display decoded image
    cv2.imshow("Decoded Image", decoded_images[0])

    # Wait for user to press a key before closing the images
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()
    cv2.imwrite("decoded_image.jpg", decoded_images[0])
