'''Image compressor based on scalar quantization and PNG.'''

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

from image_IO import image_1 as gray_image # pip install "image_IO @ git+https://github.com/vicente-gonzalez-ruiz/image_IO"
from scalar_quantization.deadzone_quantization import Deadzone_Quantizer as Quantizer # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
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
ENCODE_OUTPUT = "/tmp/reduce_graytones_encoded.png"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/reduce_graytones_decoded.png"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-q", "--QSS", type=int_or_str, help="Quantization step size", default=32)
#parser.add_argument("-d", "--decode", action="store_true", help="Decode a previously encoded image")
subparsers = parser.add_subparsers(help='You must specify one of the following subcomands:')
parser_encode = subparsers.add_parser('encode', help="Encode an image")
parser_decode = subparsers.add_parser('decode', help='Decode an image')
parser_encode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=ENCODE_OUTPUT)
parser_encode.set_defaults(func=encode)
parser_decode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=DECODE_INPUT)
parser_decode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {DECODE_OUTPUT}", default=DECODE_OUTPUT)    
parser_decode.set_defaults(func=decode)

class Reduce_Graytones:

    MIN_INDEX_VALUE = -128
    MAX_INDEX_VALUE = 127

    def __init__(self, args):
        self.args = args
        logging.info(__doc__)
        self.Q = Quantizer(Q_step=self.args.QSS, min_val=self.MIN_INDEX_VALUE, max_val=self.MAX_INDEX_VALUE)
        logging.info(f"quantizer = {quantizer_name}")

    def encode(self):
        #os.system(f"cp -f {self.args.input} /tmp/input_remove_graytones_000.png") 
        #img = gray_image.read("/tmp/input_remove_graytones_000.png", 0)
        img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        img_128 = img.astype(np.int16) - 128
        k = self.Q.encode(img_128)
        k += 128 # Only positive components can be written in a PNG file
        k = k.astype(np.uint8)
        rate = gray_image.write(k, f"/tmp/reduce_graytones_encoded_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp /tmp/reduce_graytones_encoded_000.png {self.args.output}")
        logging.info(f"Generated {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        return rate

    def decode(self):
        os.system(f"cp -f {self.args.input} /tmp/reduce_graytones_encoded_000.png") 
        k = gray_image.read("/tmp/reduce_graytones_encoded_", 0).astype(np.int16)
        k -= 128
        y = self.Q.decode(k)
        y_128 = y.astype(np.int16) + 128
        rate = gray_image.write(y_128, f"/tmp/reduce_graytones_decoded_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp /tmp/reduce_graytones_decoded_000.png {self.args.output}")
        logging.info(f"Generated {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        return rate

if __name__ == "__main__":
    parser.description = __doc__
    args = parser.parse_known_args()[0]
    print(args)

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()
    logging.info(f"QSS = {args.QSS}")

    codec = Reduce_Graytones(args)

    rate = args.func(codec)
    #if args.decode:
    #    rate = codec.decode()
    #else:
    #    rate = codec.encode()
    logging.info(f"rate = {rate} bits/pixel")

