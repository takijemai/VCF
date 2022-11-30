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
        decom_img = analyze(YCoCg_img, self.wavelet, self.levels)
        decom_k = self.quantize_decom(decom_img)
        self.save_decom(decom_k)
        rate = (self.required_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        decom_k = self.read_decom()
        decom_y = self.dequantize_decom(k)
        YCoCg_y = synthesize(decom_y, self.wavelet, self.levels)
        y_128 = to_RGB(YCoCg_y)
        y = (y_128.astype(np.int16) + 128)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.save(y)
        rate = (self.required_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

    def quantize_decom(self, d):
        k = [self.quantize(d[0])] # LL subband
        sb_counter = 1 # sb = subband
        for sr_d in d[1:]: # sr = spatial resolution
            sr_k = []
            for sb_d in sr_d:
                sb_k = self.quantize(sb_d)
                sr_k.append(sb_k)
                sb_counter += 1
        k.append(tuple(sr_k))
        return k

    def dequantize_decom(k):
        y = [self.dequantize(k[0])]
        sb_counter = 1
        for sr_k in k[1:]:
            sr_y = []
            for sb_k in sr_k: # sb = subband
                sb_y = self.dequantize(sb_k)
                sr_y.append(sb_y)
                sb_counter += 1
        y.append(tuple(sr_y))
        return y

    def read_decom(self):
        LL = io.imread(f"{prefix}LL{self.levels}")
        decom = [LL]
        resolution_index = self.levels
        for l in range(self.levels, 0, -1):
            subband_names = ["LH", "HL", "HH"]
            sb = 0
            resolution = []
            for sbn in subband_names:
                resolution.appen(io.imread(f"{self.args.input}{sbn}{resolution_index}"))
                sb += 1
            decom.append(tuple(resolution))
            resolution_index -= 1
        return decom

    def save_decom(self, decom):
        LL = decom[0]
        fn = f"{self.args.output}LL{self.levels}"
        io.imsave(fn, LL)
        self.required_bytes = os.path.getsize(fn)
        resolution_index = self.levels
        aux_decom = [decom[0][..., 0]]
        for resolution in decom[1:]:
            subband_names = ["LH", "HL", "HH"]
            sb = 0
            aux_resol = []
            for sbn in subband_names:
                fn = f"{self.args.output}{sbn}{resolution_index}"
                io.imsage(fn, resolution[sb])
                self.required_bytes = os.path.getsize(fn)
                aux_resol.append(resolution[sb][..., 0])
                sb += 1
            resolution_index -= 1
            aux_decom.append(tuple(aux_resol))
        self.slices = pywt.coeffs_to_array(aux_decom)[1]
        return slices

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

