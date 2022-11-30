'''Image quantization using a deadzone scalar quantizer'''

import argparse
import os
from skimage import io # pip install scikit-image
import numpy as np

import logging
#FORMAT = "%(module)s: %(message)s"
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT)
#logging.basicConfig(format=FORMAT, level=logging.INFO)
logging.basicConfig(format=FORMAT, level=logging.DEBUG)

# pip install "image_IO @ git+https://github.com/vicente-gonzalez-ruiz/image_IO"
#from image_IO import image_1 as gray_image
# pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
from scalar_quantization.deadzone_quantization import Deadzone_Quantizer as Quantizer
from scalar_quantization.deadzone_quantization import name as quantizer_name

import entropy

entropy.parser_encode.add_argument("-q", "--QSS", type=entropy.int_or_str, help=f"Quantization step size (default: 32)", default=32)

class Deadzone_Quantizer(entropy.Entropy_Codec):

    def __init__(self, args): # ???
        self.args = args
        logging.debug(f"args = {self.args}")

    def encode(self):
        '''Read an image, quantize the image, and save it.'''
        img = self.read()
        img_128 = img.astype(np.int16) - 128
        k, rate = self.quantize(img_128)
        rate += self.save(k)
        return rate

    def quantize(self, img, min_index_val=-128, max_index_val=127):
        '''Quantize the image.'''
        logging.info(f"QSS = {self.args.QSS}")
        self.Q = Quantizer(Q_step=self.args.QSS, min_val=min_index_val, max_val=max_index_val)
        k = self.Q.encode(img)
        k += 128 # Only positive components can be written in a PNG file
        k = k.astype(np.uint8)
        with open(f"{self.args.output}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")
        rate = 1*8/(k.shape[0]*k.shape[1]) # We suppose that the representation of the QSS requires 1 byte
        logging.info(f"Written {self.args.output}_QSS.txt")
        return k, rate

    def decode(self):
        '''Read a quantized image, "dequantize", and save.'''
        k = self.read()
        y = self.dequantize(k)
        y_128 = (y.astype(np.int16) + 128).astype(np.uint8)
        rate = self.save(y_128)
        return rate

    def dequantize(self, k, min_index_val=-128, max_index_val=127):
        '''"Dequantize" an image.'''
        k = k.astype(np.int16)
        k -= 128
        with open(f"{self.args.input}_QSS.txt", 'r') as f:
            QSS = int(f.read())
        self.Q = Quantizer(Q_step=QSS, min_val=min_index_val, max_val=max_index_val)
        logging.info(f"Read QSS={QSS} from {self.args.output}_QSS.txt")
        y = self.Q.decode(k)
        return y

if __name__ == "__main__":
    logging.info(__doc__) # ???
    logging.info(f"quantizer = {quantizer_name}")
    entropy.parser.description = __doc__
    args = entropy.parser.parse_known_args()[0]
    #args = entropy.parser.parse_args()

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = Deadzone_Quantizer(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

