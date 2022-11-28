'''Recorrelating color information with the YCoCg transform.'''

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
import color_pixel_static_scalar_quantization

# pip install "image_IO @ git+https://github.com/vicente-gonzalez-ruiz/image_IO"
from image_IO import image_1 as gray_image
# pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import from_RGB
from color_transforms.YCoCg import to_RGB

class YCoCg(color_pixel_static_scalar_quantization.Color_Pixel_Static_Scalar_Quantization):
    
    def __init__(self, args):
        super().__init__(args)

    def encode(self):
        RGB_img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {RGB_img.shape}")
        RGB_img_128 = RGB_img.astype(np.int16) - 128
        YCoCg_img = from_RGB(RGB_img_128)
        print(YCoCg_img.max(), YCoCg_img.min())
        rate = self.encode_image(YCoCg_img)
        return rate

    def decode(self):
        YCoCg_img = self.decode_image()
        print(YCoCg_img.max(), YCoCg_img.min())
        RGB_img_128 = to_RGB(YCoCg_img.astype(np.int16))
        RGB_img = (RGB_img_128 + 128)
        print(RGB_img.max(), RGB_img.min())
        rate = gray_image.write(RGB_img.astype(np.uint8), f"{gray_pixel_static_scalar_quantization.DECODE_OUTPUT}_", 0)*8/(RGB_img.shape[0]*RGB_img.shape[1])
        os.system(f"cp {gray_pixel_static_scalar_quantization.DECODE_OUTPUT}_000.png {self.args.output}")
        logging.info(f"Written {os.path.getsize(self.args.output)} bytes in {self.args.output}")
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

    codec = YCoCg(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

