'''Exploiting spatial (perceptual) redundancy with the Discrete Wavelet Transform.'''

import argparse
from skimage import io # pip install scikit-image
import numpy as np

import logging
#FORMAT = "%(module)s: %(message)s"
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT)
logging.basicConfig(format=FORMAT, level=logging.INFO)
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)

import gray_pixel_static_scalar_quantization
import color_pixel_static_scalar_quantization

# pip install "DWT @ git+https://github.com/vicente-gonzalez-ruiz/DWT"
from DWT import color_dyadic_DWT as DWT
# pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import from_RGB
from color_transforms.YCoCg import to_RGB

class DWT(YCoCg.YCoCg):
    
    def __init__(self, args):
        super().__init__(args)

    def encode(self):
        RGB_img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {RGB_img.shape}")
        RGB_img_128 = RGB_img.astype(np.int16) - 128
        YCoCg_img = from_RGB(RGB_img_128)
        rate = self.encode_image(YCoCg_img)
        return rate

    def decode(self):
        YCoCg_img = self.decode_image()
        RGB_img_128 = to_RGB(YCoCg_img.astype(np.int16))
        RGB_img = (RGB_img_128 + 128)
        RGB_img = np.clip(RGB_img, 0, 255).astype(np.uint8)
        io.imsave(self.args.output, RGB_img)
        obytes = os.path.getsize(self.args.output)
        rate = obytes*8/(RGB_img.shape[0]*RGB_img.shape[1])
        logging.info(f"Written {obytes} bytes in {self.args.output}")
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

