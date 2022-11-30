'''Entropy Encoding of images using PNG (Portable Network Graphics).'''

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

ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded.png"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(help="You must specify one of the following subcomands:", dest="subparser_name")
parser_encode = subparsers.add_parser("encode", help="Encode an image")
parser_decode = subparsers.add_parser("decode", help='Decode an image')
parser_encode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")
parser_encode.set_defaults(func=encode)
parser_decode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser_decode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {DECODE_OUTPUT}", default=f"{DECODE_OUTPUT}")    
parser_decode.set_defaults(func=decode)

class PNG_Codec:

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"encoding = {self.encoding}")

    def encode(self):
        '''Read an image and save it in the disk.'''
        # The input can be online.
        img = self.read()
        rate = self.save(img)
        return rate

    def decode(self):
        '''Read an image and save it in the disk.'''
        img = self.read()
        rate = self.save(img)
        return rate

    def read(self):
        '''Read an image.'''
        img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        return img

    def save(self, img):
        '''Save to disk the image.'''
        # The encoding algorithm depends on the output file extension.
        io.imsave(self.args.output, img)
        if __debug__:
            required_bytes = os.path.getsize(self.args.output)
        logging.info(f"Written {required_bytes} bytes in {self.args.output}")
        return required_bytes

if __name__ == "__main__":
    logging.info(__doc__) # ?
    parser.description = __doc__
    #args = parser.parse_known_args()[0]
    args = parser.parse_args()

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = PNG_Codec(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

