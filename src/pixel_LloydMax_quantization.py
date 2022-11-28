'''Pixel Domain LloydMax scalar quantization.'''

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

# pip install "image_IO @ git+https://github.com/vicente-gonzalez-ruiz/image_IO"
from image_IO import image_1 as gray_image
# pip install "scalar_quantization @ git+https://github.com/vicente-gonzalez-ruiz/scalar_quantization"
from scalar_quantization.LloydMax_quantization import LloydMax_Quantizer as Quantizer
from scalar_quantization.LloydMax_quantization import name as quantizer_name

import .gray_pixel_scalar_quantization

ENCODE_OUTPUT = "/tmp/gray_pixel_LloydMax_quantization__encoded"
DECODE_INPUT = ENCODE_OUTPUT
DECODE_OUTPUT = "/tmp/gray_pixel_LloydMax_quantization__decoded"

class Gray_Pixel_LloydMax_Quantization(Gray_Pixel_Domain_Static_Quantization):
    
    def __init__(self, args):
        logging.info(__doc__)
        super().__init__(args)

    def encode(self):
        img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        histogram_img, bin_edges_img = np.histogram(img, bins=256, range=(0, 256))
        self.Q = quantizer(Q_step=self.args.QSS, counts=histogram_img)
        k = self.Q.encode(img)
        k = k.astype(np.uint8)
        rate = gray_image.write(k, f"{ENCODE_OUTPUT}_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp {ENCODE_OUTPUT}_000.png {self.args.output}")
        logging.info(f"Generated {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        with open(f"{self.args.output}.QSS", 'w') as f:
            f.write(f"{self.args.QSS}")

        logging.info(f"Written {self.args.output}.QSS")
        return rate

        centroids = self.get_representation_levels()
        with gzip.GzipFile(f"{self.args.output}_centroids.gz", "w") as f:
            np.save(file=f, arr=centroids)

    def decode(self):
        with gzip.GzipFile(f"{self.args.input}_centroids.gz", "r") as f:
            centroids = np.load(file=f)
        self.set_representation_levels(centroids)
        super().decode()



    def __init__(self, args):
        self.args = args
        logging.info(__doc__)
        logging.debug(f"args = {self.args}")
        logging.info(f"quantizer = {quantizer_name}")

    def encode(self):
        self.Q = Quantizer(Q_step=self.args.QSS, min_val=self.MIN_INDEX_VALUE, max_val=self.MAX_INDEX_VALUE)
        img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        img_128 = img.astype(np.int16) - 128
        k = self.Q.encode(img_128)
        k += 128 # Only positive components can be written in a PNG file
        k = k.astype(np.uint8)
        rate = gray_image.write(k, f"/tmp/reduce_graytones_encoded_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp /tmp/reduce_graytones_encoded_000.png {self.args.output}")
        logging.info(f"Generated {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        with open(f"{self.args.output}.QSS", 'w') as f:
            f.write(f"{self.args.QSS}")
        logging.info(f"Written {self.args.output}.QSS")
        return rate

    def decode(self):
        with open(f"{self.args.input}.QSS", 'r') as f:
            QSS = int(f.read())
        self.Q = Quantizer(Q_step=QSS, min_val=self.MIN_INDEX_VALUE, max_val=self.MAX_INDEX_VALUE)
        logging.info(f"Read {QSS} from {self.args.output}.QSS")
        os.system(f"cp -f {self.args.input} /tmp/reduce_graytones_encoded_000.png") 
        k = gray_image.read("/tmp/reduce_graytones_encoded_", 0).astype(np.int16)
        k -= 128
        y = self.Q.decode(k)
        y_128 = y.astype(np.int16) + 128
        rate = gray_image.write(y_128, f"/tmp/reduce_graytones_decoded_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp /tmp/reduce_graytones_decoded_000.png {self.args.output}")
        logging.info(f"Generated {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        return rate

if __name__ == "__main__":
    parser.description = __doc__
    args = parser.parse_known_args()[0]
    #args = parser.parse_args()[0]
    print(args)

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = Reduce_Graytones(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

