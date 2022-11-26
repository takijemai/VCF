'''Image compressor based on scalar quantization and PNG.'''

import argparse
import logging
import os
from image_IO import image_1 as gray_image # pip install "image_IO @ git+https://github.com/vicente-gonzalez-ruiz/image_IO"
from scalar_quantization import deadzone_quantization as quantizer # pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"

#FORMAT = "%(module)s: %(message)s"
FORMAT = "(%(levelname)s) %(module)s: %(message)s"
#logging.basicConfig(format=FORMAT)
logging.basicConfig(format=FORMAT, level=logging.INFO)

def int_or_str(text):
    '''Helper function for argument parsing.'''
    try:
        return int(text)
    except ValueError:
        return text

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("-i", "--input", type=int_or_str, help="Input image")
parser.add_argument("-o", "--output", type=int_or_str, help="Output image")
parser.add_argument("-q", "--QSS", type=int_or_str, help="Quantization step size")
parser.add_argument("-d", "--decode", action="store_true", help="Decode")

class Remove_Graytones:
    def __init__(self, args):
        self.args = args
        logging.info(__doc__)

    def encode(self):
        os.system(f"cp {self.args.input} /tmp/input_remove_graytones_000.png") 
        img = gray_image.read("/tmp/input_remove_graytones_000.png", 0)
        img_128 = img.astype(np.int16) - 128
        QSS = 16 # Quantization Step Size
        Q = quantizer(Q_step=self.args.QSS, min_val=-128, max_val=127)
        y, k = Q.encode_decode(img_128)
        k += 128 # Only positive components can be written in a PNG file
        k = k.astype(np.uint8)
        rate = gray_image.write(k, f"/tmp/output_remove_graytones_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp /tmp/output_remove_graytones_000.png {self.args.output}")
        return rate

    def decode(self):
        os.system(f"cp {self.args.input} /tmp/input_remove_graytones_000.png") 
        img = gray_image.read("/tmp/input_remove_graytones_000.png", 0)
        img_128 = img.astype(np.int16) - 128
        QSS = 16 # Quantization Step Size
        Q = quantizer(Q_step=self.args.QSS, min_val=-128, max_val=127)
        y, k = Q.encode_decode(img_128)
        k += 128 # Only positive components can be written in a PNG file
        k = k.astype(np.uint8)
        rate = gray_image.write(k, f"/tmp/output_remove_graytones_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp /tmp/output_remove_graytones_000.png {self.args.output}")
        return rate

if __name__ == "__main__":
    parser.description = __doc__
    try:
        argcomplete.autocomplete(parser)
    except Exception:
        logging.warning("argcomplete not working :-/")
    args = parser.parse_known_args()[0]

    codec = Remove_Graytones(args)

