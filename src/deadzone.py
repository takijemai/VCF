'''Gray Pixel Static Scalar Quantization.'''

import argparse
import os
from skimage import io # pip install scikit-image
import numpy as np
import logging
#FORMAT = "%(module)s: %(message)s"
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT)
logging.basicConfig(format=FORMAT, level=logging.INFO)
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)

# pip install "image_IO @ git+https://github.com/vicente-gonzalez-ruiz/image_IO"
#from image_IO import image_1 as gray_image
# pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
from scalar_quantization.deadzone_quantization import Deadzone_Quantizer as Quantizer
from scalar_quantization.deadzone_quantization import name as quantizer_name

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

def encode(codec):
    return codec.encode()

def decode(codec):
    return codec.decode()

ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/barb.png"
ENCODE_OUTPUT = "/tmp/encoded.png"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(help='You must specify one of the following subcomands:')
parser_encode = subparsers.add_parser('encode', help="Encode a gray-scaled image")
parser_decode = subparsers.add_parser('decode', help='Decode a gray-scaled image')
parser_encode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")
parser_encode.add_argument("-q", "--QSS", type=int_or_str, help=f"Quantization step size (default: 32)", default=32)
parser_encode.set_defaults(func=encode)
parser_decode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser_decode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {DECODE_OUTPUT}", default=f"{DECODE_OUTPUT}")    
parser_decode.set_defaults(func=decode)

class Gray_Pixel_Static_Scalar_Quantization:

    #MIN_INDEX_VALUE = -128
    #MAX_INDEX_VALUE = 127

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")

    def encode_image(self, img, min_index_val=-128, max_index_val=127):
        logging.info(f"QSS = {self.args.QSS}")
        self.Q = Quantizer(Q_step=self.args.QSS, min_val=min_index_val, max_val=max_index_val)
        k = self.Q.encode(img)
        k += 128 # Only positive components can be written in a PNG file
        k = k.astype(np.uint8)
        io.imsave(self.args.output, k)
        obytes = os.path.getsize(self.args.output)
        rate = obytes*8/(k.shape[0]*k.shape[1])
        logging.info(f"Written {obytes} bytes in {self.args.output}.png")
        with open(f"{self.args.output}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")
        rate += 1*8/(k.shape[0]*k.shape[1]) # We suppose that the representation of the QSS requires 1 byte
        logging.info(f"Written {self.args.output}_QSS.txt")
        return rate

    def encode(self):
        img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        img_128 = img.astype(np.int16) - 128
        rate = self.encode_image(img_128)
        return rate

    def decode_image(self, min_index_val=-128, max_index_val=127):
        with open(f"{self.args.input}_QSS.txt", 'r') as f:
            QSS = int(f.read())
        self.Q = Quantizer(Q_step=QSS, min_val=min_index_val, max_val=max_index_val)
        logging.info(f"Read QSS={QSS} from {self.args.output}_QSS.txt")
        k = io.imread(self.args.input).astype(np.int16)
        k -= 128
        y = self.Q.decode(k)
        return y

    def decode(self):
        y = self.decode_image()
        y_128 = (y.astype(np.int16) + 128).astype(np.uint8)
        io.imsave(self.args.output, y_128)
        obytes = os.path.getsize(self.args.output)
        rate = obytes*8/(y_128.shape[0]*y_128.shape[1])
        logging.info(f"Written {obytes} bytes in {self.args.output}")
        return rate

if __name__ == "__main__":
    logging.info(__doc__)
    logging.info(f"quantizer = {quantizer_name}")
    parser.description = __doc__
    args = parser.parse_known_args()[0]

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = Gray_Pixel_Static_Scalar_Quantization(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

