'''Entropy Encoding of images using PNG (Portable Network Graphics).'''

import argparse
import os
from skimage import io  # pip install scikit-image
from PIL import Image  # pip install
import numpy as np
import logging
import subprocess
import cv2 as cv
import main
import urllib
import zlib
from sklearn.linear_model import LinearRegression
from scipy.interpolate import NearestNDInterpolator


def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

# A way of converting a call to a object's method to a plain function


def predict(self, data):
    # Predict the value of each pixel based on the values of its neighbors
    # using a linear regression model
    model = LinearRegression()
    # Fit the model to the data
    model.fit(data, data[:, :, 1])
    # Predict the values for the remaining channels
    for i in range(data.shape[2]):
        data[:, :, i] = model.predict(data)
    return data


def compress(self, data):
    # Predict the image data
    predicted_data = self.predict(data)
    # Compress the predicted data using zlib
    return zlib.compress(predicted_data, level=COMPRESSION_LEVEL)


def encode(codec):
    return codec.encode()


def decode(codec):
    return codec.decode()


# Default IO images
ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"
ENCODE_OUTPUT = r'C:\Users\Usuario\OneDrive\Bureau\Project VCF\env/encoded.png'
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = r'C:\Users\Usuario\OneDrive\Bureau\Project VCF\env/decoded.png'

# Main parameter of the arguments parser: "encode" or "decode"
parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-g", "--debug", action="store_true",
                    help=f"Output debug information")
subparsers = parser.add_subparsers(
    help="You must specify one of the following subcomands:", dest="subparser_name")

# Encoder parser
parser_encode = subparsers.add_parser("encode", help="Encode an image")
parser_encode.add_argument("-i", "--input", type=int_or_str,
                           help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode.add_argument("-o", "--output", type=int_or_str,
                           help=f"Output image (default: {ENCODE_OUTPUT})", default=f"{ENCODE_OUTPUT}")
parser_encode.set_defaults(func=encode)

# Decoder parser
parser_decode = subparsers.add_parser("decode", help='Decode an image')
parser_decode.add_argument("-i", "--input", type=int_or_str,
                           help=f"Input image (default: {DECODE_INPUT})", default=f"{DECODE_INPUT}")
parser_decode.add_argument("-o", "--output", type=int_or_str,
                           help=f"Output image (default: {DECODE_OUTPUT})", default=f"{DECODE_OUTPUT}")
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
        self.input_bytes = 0
        self.output_bytes = 0

    def read_fn(self, fn):
        '''Read the image <fn>.'''
        # img = io.imread(fn) # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
        # img = Image.open(fn) # https://pillow.readthedocs.io/en/stable/handbook/tutorial.html#using-the-image-class
        try:
            input_size = os.path.getsize(fn)
            self.input_bytes += input_size
            img = cv.imread(fn, cv.IMREAD_UNCHANGED)
            img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        except:
            req = urllib.request.Request(fn, method='HEAD')
            f = urllib.request.urlopen(req)
            input_size = int(f.headers['Content-Length'])
            self.input_bytes += input_size
            # https://scikit-image.org/docs/stable/api/skimage.io.html#skimage.io.imread
            img = io.imread(fn)
        logging.info(
            f"Read {input_size} bytes from {fn} with shape {img.shape} and type={img.dtype}")
        return img

    def _write_fn(self, img, fn):
        '''Write to disk the image with filename <fn>.'''
        # Notice that the encoding algorithm depends on the output
        # file extension (PNG).
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])
        # if __debug__:
        #    len_output = os.path.getsize(fn)
        #    logging.info(f"Before optipng: {len_output} bytes")
        #subprocess.run(f"optipng {fn}", shell=True, capture_output=True)
        self.output_bytes += os.path.getsize(fn)
        logging.info(
            f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

    def write_fn(self, img, fn):
        '''Write to disk the image with filename <fn>.'''
        # Notice that the encoding algorithm depends on the output
        # file extension (PNG).
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        cv.imwrite(fn, img, [cv.IMWRITE_PNG_COMPRESSION, COMPRESSION_LEVEL])

        #io.imsave(fn, img, check_contrast=False)
        #image = Image.fromarray(img.astype('uint8'), 'RGB')
        # image.save(fn)
        #subprocess.run(f"optipng -nc {fn}", shell=True, capture_output=True)
        subprocess.run(f"pngcrush {fn} /tmp/pngcrush.png",
                       shell=True, capture_output=True)
        subprocess.run(
            f"mv -f /tmp/pngcrush.png {fn}", shell=True, capture_output=True)
        self.output_bytes += os.path.getsize(fn)
        logging.info(
            f"Written {os.path.getsize(fn)} bytes in {fn} with shape {img.shape} and type {img.dtype}")

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
        img = self.read_fn(self.args.input)
        # Compress the image data using zlib
        compressed_data = self.compress(img)
        # Write the compressed data to the output image
        self.write_fn(self.args.output, compressed_data)
        self.write(img)
        logging.debug(
            f"output_bytes={self.output_bytes}, img.shape={img.shape}")
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        '''Read an image and save it in the disk. Notice that we are
        using the PNG image format for both, decode and encode an
        image. For this reason, both methods do exactly the same.
        This method is overriden in child classes.

        '''
        return self.encode()

    def __del__(self):
        logging.info(f"Total {self.input_bytes} bytes read")
        logging.info(f"Total {self.output_bytes} bytes written")


if __name__ == "__main__":
    main.main(parser, logging, CoDec)
