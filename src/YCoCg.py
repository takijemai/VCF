import argparse
from skimage import io  # pip install scikit-image
import numpy as np
import logging
import main

import PNG as EC
import deadzone as Q
import LloydMax as LM

# pip install "color_transforms @ git+https://github.com/vicente-gonzalez-ruiz/color_transforms"
from color_transforms.YCoCg import from_RGB
from color_transforms.YCoCg import to_RGB


class CoDec(Q.CoDec):

    def __init__(self, input_file, output_file, QSS, quantizer):
        self.input_file = input_file
        self.output_file = output_file
        self.QSS = QSS
        self.quantizer = quantizer
        self.input_bytes = 0
        self.output_bytes = 0

    def quantize(self, img):
        if self.quantizer == "deadzone":
            return Q.quantize(img, self.QSS)
        elif self.quantizer == "lloydmax":
            return LM.quantize(img, self.QSS)
        else:
            raise ValueError("Invalid quantizer: {}".format(self.quantizer))

    def dequantize(self, k):
        if self.quantizer == "deadzone":
            return Q.dequantize(k, self.QSS)
        elif self.quantizer == "lloydmax":
            return LM.dequantize(k, self.QSS)
        else:
            raise ValueError("Invalid quantizer: {}".format(self.quantizer))

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
        y_128 = to_RGB(YCoCg_y)
        y = (y_128.astype(np.int16) + 128)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Exploiting color (perceptual) redundancy with the YCoCg transform.')
    parser.add_argument(
        '-i', '--input', default='http://www.hpca.ual.es/~vruiz/images/lena.png', help='Input image')
    parser.add_argument(
        '-o', '--output', default=r'C:\Users\Usuario\OneDrive\Bureau\Project VCF\env/encodedY.png', help='Output image')
    parser.add_argument('-q', '--QSS', type=int, default=32,
                        help='Quantization step size (default: 32)')
    parser.add_argument('-Q', '--Quantizer', type=str,
                        default='deadzone', help='Name of the quantizer (default: deadzone)')
    args = parser.parse_args()
    main.main(EC.parser, logging, CoDec)
