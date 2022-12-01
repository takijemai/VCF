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
from DWT.color_dyadic_DWT import analyze as DWT_analyze
from DWT.color_dyadic_DWT import synthesize as DWT_synthesize

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
        decom_img = DWT_analyze(YCoCg_img, self.wavelet, self.levels)
        decom_k = self.quantize_decom(decom_img)
        self.save_decom(decom_k)
        rate = (self.required_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        decom_k = self.read_decom()
        decom_y = self.dequantize_decom(k)
        YCoCg_y = DWT_synthesize(decom_y, self.wavelet, self.levels)
        y_128 = to_RGB(YCoCg_y)
        y = (y_128.astype(np.int16) + 128)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.save(y)
        rate = (self.required_bytes*8)/(k.shape[0]*k.shape[1])
        return rate

    def quantize_decom(self, decom):
        decom_k = [self.quantize(decom[0])] # LL subband
        for spatial_resolution in decom[1:]:
            spatial_resolution_k = []
            for subband in spatial_resolution:
                subband_k = self.quantize(subband)
                spatial_resolution_k.append(subband_k)
        decom_k.append(tuple(spatial_resolution_k))
        return decom_k

    def dequantize_decom(decom_k):
        decom_y = [self.dequantize(decom_k[0])]
        for spatial_resolution_k in decom_k[1:]:
            spatial_resolution_y = []
            for subband_k in spatial_resolution_k:
                subband_y = self.dequantize(subband_k)
                spatial_resolution_y.append(subband_y)
        decom_y.append(tuple(spatial_resolution_y))
        return decom_y

    def read_decom(self):
        LL = self.read(f"{prefix}LL{self.levels}")
        decom = [LL]
        resolution_index = self.levels
        for l in range(self.levels, 0, -1):
            subband_names = ["LH", "HL", "HH"]
            spatial_resolution = []
            for subband_name in subband_names:
                spatial_resolution.appen(self.read(f"{self.args.input}{subband_name}{resolution_index}"))
            decom.append(tuple(resolution))
            resolution_index -= 1
        return decom

    def save_decom(self, decom):
        LL = decom[0]
        fn = f"{self.args.output}LL{self.levels}"
        print(LL.dtype)
        self.save_fn(LL, fn)
        resolution_index = self.levels
        aux_decom = [decom[0][..., 0]] # Used for computing slices
        for spatial_resolution in decom[1:]:
            subband_names = ["LH", "HL", "HH"]
            subband_index = 0
            aux_resol = [] # Used for computing slices
            for subband_name in subband_names:
                fn = f"{self.args.output}{subband_name}{resolution_index}"
                self.save_fn(spatial_resolution[subband_index], fn)
                aux_resol.append(spatial_resolution[sb][..., 0])
                subband_index += 1
            resolution_index -= 1
            aux_decom.append(tuple(aux_resol))
        self.slices = pywt.coeffs_to_array(aux_decom)[1]
        return slices

    def save_fn(self, img, fn):
        print(fn)
        io.imsave(fn, img)
        self.required_bytes = os.path.getsize(fn)
        logging.info(f"Written {self.required_bytes} bytes in {self.args.output}")

    def read_fn(self, fn):
        img = io.imread(fn)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        return img

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

