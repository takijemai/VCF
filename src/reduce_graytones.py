'''Image compressor based on scalar quantization and PNG.'''

import argparse
import os
from skimage import io # pip install scikit-image
import numpy as np
import logging
#FORMAT = "%(module)s: %(message)s"
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT)
logging.basicConfig(format=FORMAT, level=logging.INFO)
#logging.basicConfig(format=FORMAT, level=logging.DEBUG)

from image_IO import image_1 as gray_image # pip install "image_IO @ git+https://github.com/vicente-gonzalez-ruiz/image_IO"
from scalar_quantization.deadzone_quantization import Deadzone_Quantizer as Quantizer # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", type=int_or_str, help="Input image", default="http://www.hpca.ual.es/~vruiz/images/barb.png")
parser.add_argument("-o", "--output", type=int_or_str, help="Output image", default="/tmp/reduce_graytones.png")
parser.add_argument("-q", "--QSS", type=int_or_str, help="Quantization step size", default=32)
parser.add_argument("-d", "--decode", action="store_true", help="Decode a previously encoded image")

class Reduce_Graytones:

    MIN_INDEX_VALUE = -128
    MAX_INDEX_VALUE = 127

    def __init__(self, args):
        self.args = args
        logging.info(__doc__)
        self.Q = Quantizer(Q_step=self.args.QSS, min_val=self.MIN_INDEX_VALUE, max_val=self.MAX_INDEX_VALUE)

    def encode(self):
        #os.system(f"cp -f {self.args.input} /tmp/input_remove_graytones_000.png") 
        #img = gray_image.read("/tmp/input_remove_graytones_000.png", 0)
        img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        img_128 = img.astype(np.int16) - 128
        k = self.Q.encode(img_128)
        k += 128 # Only positive components can be written in a PNG file
        k = k.astype(np.uint8)
        rate = gray_image.write(k, f"/tmp/reduce_graytones_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp /tmp/reduce_graytones_000.png {self.args.output}")
        logging.info(f"Generated {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        return rate

    def decode(self):
        os.system(f"cp -f {self.args.input} /tmp/reduce_graytones_000.png") 
        k = gray_image.read("/tmp/reduce_graytones_", 0)
        k += 128
        y = self.Q.decode(k)
        rate = gray_image.write(y, f"/tmp/reduce_graytones_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp /tmp/reduce_graytones_000.png {self.args.output}")
        return rate

if __name__ == "__main__":
    parser.description = __doc__
    args = parser.parse_known_args()[0]

    logging.info(f"input = {args.input}")
    logging.info(f"output = {args.output}")
    logging.info(f"QSS = {args.QSS}")

    codec = Reduce_Graytones(args)
    if args.decode:
        rate = codec.decode()
    else:
        rate = codec.encode()
    logging.info(f"rate = {rate}")

