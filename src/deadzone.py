'''Image quantization using a deadzone scalar quantizer'''

import argparse
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

import PNG as EC # Entropy Coding

EC.parser_encode.add_argument("-q", "--QSS", type=EC.int_or_str, help=f"Quantization step size (default: 32)", default=32)

class CoDec(EC.CoDec):

    def __init__(self, args, min_index_val=-128, max_index_val=127): # ???
        super().__init__(args)
        logging.debug(f"args = {self.args}")
        if self.encoding:
            self.QSS = args.QSS
            logging.info(f"QSS = {self.QSS}")
            with open(f"{args.output}_QSS.txt", 'w') as f:
                f.write(f"{self.args.QSS}")
                logging.info(f"Written {self.args.output}_QSS.txt")
        else:
            with open(f"{args.input}_QSS.txt", 'r') as f:
                self.QSS = int(f.read())
                logging.info(f"Read QSS={self.QSS} from {self.args.output}_QSS.txt")
        self.Q = Quantizer(Q_step=self.QSS, min_val=min_index_val, max_val=max_index_val)
        self.required_bytes = 1 # We suppose that the representation of QSS requires 1 byte.

    def encode(self):
        '''Read an image, quantize the image, and save it.'''
        img = self.read()
        img_128 = img.astype(np.int16) - 128
        k = self.quantize(img_128)
        self.save(k)
        rate = (self.required_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def quantize(self, img):
        '''Quantize the image.'''
        k = self.Q.encode(img)
        k += 128 # Only positive components can be written in a PNG file
        k = k.astype(np.uint8)
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")
        return k

    def decode(self):
        '''Read a quantized image, "dequantize", and save.'''
        k = self.read()
        y_128 = self.dequantize(k)
        y = (np.rint(y_128).astype(np.int16) + 128).astype(np.uint8)
        logging.debug(f"y.shape={y.shape} y.dtype={y.dtype}")        
        self.save(y)
        rate = (self.required_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

    def dequantize(self, k):
        '''"Dequantize" an image.'''
        k = k.astype(np.int16)
        k -= 128
        logging.debug(f"k.shape={k.shape} k.dtype={k.dtype}")
        #self.Q = Quantizer(Q_step=QSS, min_val=min_index_val, max_val=max_index_val)
        y = self.Q.decode(k)
        logging.debug(f"y.shape={y.shape} y.dtype={y.dtype}")
        assert y.all() > -129
        assert y.all() < 128
        return y

if __name__ == "__main__":
    logging.info(__doc__) # ???
    logging.info(f"quantizer = {quantizer_name}")
    EC.parser.description = __doc__
    args = EC.parser.parse_known_args()[0]
    #args = EC.parser.parse_args()

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = CoDec(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

