'''Color Pixel Static Scalar Quantization.'''

import argparse
import os
from skimage import io # pip install scikit-image
import numpy as np
import gzip

import logging
#FORMAT = "%(module)s: %(message)s"
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT)
logging.basicConfig(format=FORMAT, level=logging.INFO)
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)

import gray_pixel_static_scalar_quantization

ENCODE_INPUT = "http://www.hpca.ual.es/~vruiz/images/lena.png"

parser_encode_color = gray_pixel_static_scalar_quantization.subparsers.add_parser('encode_color', help="Encode a color image")
parser_decode_color = gray_pixel_static_scalar_quantization.subparsers.add_parser('decode_color', help="Decode a color image")
parser_encode_color.add_argument("-i", "--input", type=gray_pixel_static_scalar_quantization.int_or_str, help=f"Input image (default: {ENCODE_INPUT})", default=ENCODE_INPUT)
parser_encode_color.add_argument("-o", "--output", type=gray_pixel_static_scalar_quantization.int_or_str, help=f"Output image (default: {gray_pixel_static_scalar_quantization.ENCODE_OUTPUT})", default=f"{gray_pixel_static_scalar_quantization.ENCODE_OUTPUT}")
parser_encode_color.add_argument("-q", "--QSS", type=gray_pixel_static_scalar_quantization.int_or_str, help=f"Quantization step size (default: 32)", default=32)
parser_encode_color.set_defaults(func=gray_pixel_static_scalar_quantization.encode)
parser_decode_color.add_argument("-i", "--input", type=gray_pixel_static_scalar_quantization.int_or_str, help=f"Input image (default: {gray_pixel_static_scalar_quantization.DECODE_INPUT})", default=f"{gray_pixel_static_scalar_quantization.DECODE_INPUT}")
parser_decode_color.add_argument("-o", "--output", type=gray_pixel_static_scalar_quantization.int_or_str, help=f"Output image (default: {gray_pixel_static_scalar_quantization.DECODE_OUTPUT}", default=f"{gray_pixel_static_scalar_quantization.DECODE_OUTPUT}")    
parser_decode_color.set_defaults(func=gray_pixel_static_scalar_quantization.decode)

class Color_Pixel_LloydMax_Quantization(gray_pixel_static_scalar_quantization.Gray_Pixel_Static_Scalar_Quantization):
    
    def __init__(self, args):
        super().__init__(args)

    def encode(self):
        rate = super().encode()
        return rate

    def decode(self):
        rate = super().decode()
        return rate

if __name__ == "__main__":
    logging.info(__doc__)
    logging.info(f"quantizer = {gray_pixel_static_scalar_quantization.quantizer_name}")
    gray_pixel_static_scalar_quantization.parser.description = __doc__
    args = gray_pixel_static_scalar_quantization.parser.parse_known_args()[0]

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = Color_Pixel_LloydMax_Quantization(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

