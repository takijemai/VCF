'''Exploiting color (perceptual) redundancy with the YCoCg transform.'''

import argparse
from skimage import io  # pip install scikit-image
import numpy as np
import logging
import main

import PNG as EC
import deadzone as DZ
import LloydMax as LM

# pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import from_RGB
from color_transforms.YCoCg import to_RGB


class CoDec(object):
    def __init__(self, method):
        self.method = method
        if method == "deadzone":
            self.quantize = DZ.quantize
            self.dequantize = DZ.dequantize
        elif method == "LloydMax":
            self.quantize = LM.quantize
            self.dequantize = LM.dequantize
        else:
            raise ValueError("Invalid quantization method")

    def encode(self):
        img = self.read()
        img_128 = img.astype(np.int16) - 128
        YCoCg_img = from_RGB(img_128)
        k = self.quantize(YCoCg_img)
        self.write(k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        k = self.read()
        YCoCg_y = self.dequantize(k)
        #y_128 = to_RGB(YCoCg_y.astype(np.int16))
        y_128 = to_RGB(YCoCg_y)
        y = (y_128.astype(np.int16) + 128)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--method", choices=["deadzone", "LloydMax"],
                        required=True, help="Quantization method to use")
    args = parser.parse_args()
    main.main(EC.parser, logging, CoDec, args.method)
