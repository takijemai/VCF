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
# new command-line argument to specify the quantization method
parser = argparse.ArgumentParser(
    description="Exploiting color (perceptual) redundancy with the YCoCg transform.")
parser.add_argument("--quantization", type=str, choices=[
                    "deadzone", "lloydmax"], help="Choose the quantization method: deadzone or lloydmax")


class CoDec(DZ.CoDec):
    def __init__(self, args):
        self.quantization = args.quantization
        super().__init__(args)

    def encode(self):
        img = self.read()
        img_128 = img.astype(np.int16) - 128
        YCoCg_img = from_RGB(img_128)
        if self.quantization == 'deadzone':
            k = self.quantize(YCoCg_img)
        elif self.quantization == 'Lloydmax':
            self.quantize = LM.quantize
            k = LM.quantize(YCoCg_img)
        self.write(k)
        rate = (self.output_bytes*8)/(img.shape[0]*img.shape[1])
        return rate

    def decode(self):
        k = self.read()
        if self.quantization == "deadzone":
            YCoCg_y = self.dequantize(k)
        elif self.quantization == "lloydmax":
            self.dequantize = LM.dequantize
            YCoCg_y = LM.dequantize(k)
        #y_128 = to_RGB(YCoCg_y.astype(np.int16))
        y_128 = to_RGB(YCoCg_y)
        y = (y_128.astype(np.int16) + 128)
        y = np.clip(y, 0, 255).astype(np.uint8)
        self.write(y)
        rate = (self.input_bytes*8)/(k.shape[0]*k.shape[1])
        return rate


if __name__ == "__main__":
    args = parser.parse_args()
    main.main(EC.parser, logging, CoDec)
