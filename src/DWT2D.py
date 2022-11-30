'''Exploiting spatial (perceptual) redundancy with the 2D dyadic Discrete Wavelet Transform.'''

import argparse
from skimage import io # pip install scikit-image
import numpy as np
import pywt

import logging
#FORMAT = "%(module)s: %(message)s"
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT)
logging.basicConfig(format=FORMAT, level=logging.INFO)
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)

import PNG
import YCoCg

# pip install "DWT @ git+https://github.com/vicente-gonzalez-ruiz/DWT"
#from DWT import color_dyadic_DWT as DWT
from DWT.color_dyadic_DWT import analyze

# pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import from_RGB
from color_transforms.YCoCg import to_RGB

PNG.parser_encode.add_argument("-l", "--levels", type=PNG.int_or_str, help=f"Number of decomposition levels (default: 5)", default=5)
PNG.parser_encode.add_argument("-w", "--wavelet", type=PNG.int_or_str, help=f"Wavelet name (default: \"db5\")", default="db5")

class DWT2D(YCoCg.YCoCg):

    def __init__(self, args):
        super().__init__(args)
        self.levels = args.levels
        self.wavelet = pywt.Wavelet(args.wavelet)

    def encode(self):
        img = self.read()
        img_128 = img.astype(np.int16) - 128
        YCoCg_img = from_RGB(img_128)
        decom_img = analyze(YCoCg_img, self.wavelet, self.args.levels)
        decom_k = self.quantize_decom(decom_img)
        self.required_bytes += self.save_decom(k_decom)
        rate = (self.required_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def quantize_decom(self, decom):
        LL = decom[0]
        LL_k = self.quantize(LL)
        decom_k = [add_offset(LL_k)]
        sbc_counter = 1
        for sr in decom[1:]:
            sr_k = []
            for sb in sr: # sb = subband
                Q = Quantizer(Q_step=sb_Q_factor)
            sb_k = Q.quantize(sb)
            sb_dQ = Q.dequantize(sb_k)
            sr_k.append(add_offset(sb_k))
            sr_dQ.append(sb_dQ)
            sbc_counter += 1
        decom_k.append(tuple(sr_k))
        decom_dQ.append(tuple(sr_dQ))

    def decode(self):
        k = self.read()
        DWT_y = self.dequantize(k)
        YCoCg_y = synthesize(DWT_y, self.wavelet, self.args.levels)
        y_128 = to_RGB(YCoCg_y)
        y = (y_128.astype(np.int16) + 128)
        y = np.clip(y, 0, 255).astype(np.uint8)
        rate = self.save(y)
        return rate

if __name__ == "__main__":
    logging.info(__doc__)
    #logging.info(f"quantizer = {gray_pixel_static_scalar_quantization.quantizer_name}")
    PNG.parser.description = __doc__
    args = PNG.parser.parse_known_args()[0]

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = DWT2D(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

