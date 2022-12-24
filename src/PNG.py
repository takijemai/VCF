'''Entropy Encoding of images using PNG (Portable Network Graphics).'''

import argparse
import os
from skimage import io # pip install scikit-image
from PIL import Image # pip install 
import numpy as np
import logging
import subprocess
import cv2 as cv
import main

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

# A way of converting a call to a object's method to a plain function
def encode(codec):
    return codec.encode()

def decode(codec):
    return codec.decode()

# Default IO images
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = "/tmp/encoded.png"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/decoded.png"

# Main parameter of the arguments parser: "encode" or "decode"
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g", "--debug", action="store_true", help=f"Output debug information")
subparsers = parser.add_subparsers(help="You must specify one of the following subcomands:", dest="subparser_name")

# Encoder parser
parser_encode = subparsers.add_parser("encode", help="Encode an image")
parser_encode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")
parser_encode.set_defaults(func=encode)

# Decoder parser
parser_decode = subparsers.add_parser("decode", help='Decode an image')
parser_decode.add_argument("-i", "--input", type=int_or_str, help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser_decode.add_argument("-o", "--output", type=int_or_str, help=f"Output image (default: {DECODE_OUTPUT}", default=f"{DECODE_OUTPUT}")    
parser_decode.set_defaults(func=decode)

COMPRESSION_LEVEL = 9

class CoDec:

    def __init__(self, args):
        self.args = args
        logging.debug(f"args = {self.args}")
        if args.subparser_name == "encode":
            self.encoding = True
        else:
            self.encoding = False
        logging.debug(f"encoding = {self.encoding}")
        self.required_bytes = 0 # Number of output bytes

    def read_fn(self, fn):
        '''Read the image <fn>.'''
        img = io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        #img = Image.open(fn) # https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#using-the-image-class
        logging.info(f"Read {fn} of shape {img.shape}")
        logging.debug(f"img.shape={img.shape} img.dtype={img.dtype}")
        return img

    def _write_fn(self, img, fn):
        '''Write to disk the image with filename <fn>.'''
        # Notice that the encoding algorithm depends on the output
        # file extension (PNG).
        logging.debug(f"img.shape={img.shape} img.dtype={img.dtype}")
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])
        #if __debug__:
        #    len_output = os.path.getsize(fn)
        #    logging.info(f"Before optipng: {len_output} bytes")
        #subprocess.run(f"optipng {fn}", shell=True, capture_output=True)
        self.required_bytes = os.path.getsize(fn)
        logging.info(f"Written {self.required_bytes} bytes in {fn}")

    def write_fn(self, img, fn):
        '''Write to disk the image with filename <fn>.'''
        # Notice that the encoding algorithm depends on the output
        # file extension (PNG).
        logging.debug(f"img.shape={img.shape} img.dtype={img.dtype}")
        io.imsave(fn, img, check_contrast=False)
        #image = Image.fromarray(img.astype('uint8'), 'RGB')
        #image.save(fn)
        #subprocess.run(f"optipng -nc {fn}", shell=True, capture_output=True)
        subprocess.run(f"pngcrush {fn} /tmp/pngcrush.png", shell=True, capture_output=True)
        subprocess.run(f"mv -f /tmp/pngcrush.png {fn}", shell=True, capture_output=True)
        self.required_bytes = os.path.getsize(fn)
        logging.info(f"Written {self.required_bytes} bytes in {fn}")

    def read(self):
        '''Read the image specified in the class attribute
        <args.input>.'''
        return self.read_fn(self.args.input)

    def write(self, img):
        '''Save to disk the image specified in the class attribute <
        args.output>.'''
        self.write_fn(img, self.args.output)
        
    def encode(self):
        '''Read an image and save it in the disk. The input can be
        online. This method is overriden in child classes.'''
        img = self.read()
        self.write(img)
        rate = (self.required_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        '''Read an image and save it in the disk. Notice that we are
        using the PNG image format for both, decode and encode an
        image. For this reason, both methods do exactly the same.
        This method is overriden in child classes.

        '''
        return self.encode()

if __name__ == "__main__":
    main.main(parser, logging, CoDec)
