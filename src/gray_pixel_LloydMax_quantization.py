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

import gray_pixel_static_scalar_quantization

class Gray_Pixel_LloydMax_Quantization(gray_pixel_static_scalar_quantization.Gray_Pixel_Static_Scalar_Quantization):
    
    def __init__(self, args):
        super().__init__(args)

    def encode(self):
        logging.info(f"QSS = {self.args.QSS}")
        img = io.imread(self.args.input)
        logging.info(f"Read {self.args.input} of shape {img.shape}")
        histogram_img, bin_edges_img = np.histogram(img, bins=256, range=(0, 256))
        self.Q = Quantizer(Q_step=self.args.QSS, counts=histogram_img)
        k = self.Q.encode(img)
        k = k.astype(np.uint8)
        rate = gray_image.write(k, f"{gray_pixel_static_scalar_quantization.ENCODE_OUTPUT}_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp {gray_pixel_static_scalar_quantization.ENCODE_OUTPUT}_000.png {self.args.output}")
        logging.info(f"Written {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        centroids = self.Q.get_representation_levels()
        with gzip.GzipFile(f"{self.args.output}_centroids.gz", "w") as f:
            np.save(file=f, arr=centroids)
        len_codebook = os.path.getsize(f"{self.args.output}_centroids.gz")
        logging.info(f"Written {len_codebook} bytes in {self.args.output}_centroids.gz")
        rate += len_codebook/8/(k.shape[0]*k.shape[1])
        with open(f"{self.args.output}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")
        rate += 1*8/(k.shape[0]*k.shape[1]) # We suppose that the representation of the QSS requires 1 byte
        logging.info(f"Written {self.args.output}_QSS.txt")
        return rate

    def decode(self):
        with open(f"{self.args.input}_QSS.txt", 'r') as f:
            QSS = int(f.read())
        logging.info(f"Read QSS={QSS} from {self.args.output}_QSS.txt")
        with gzip.GzipFile(f"{self.args.input}_centroids.gz", "r") as f:
            centroids = np.load(file=f)
        logging.info(f"Rea {self.args.input}_centroids.gz")
        self.Q = Quantizer(Q_step=QSS, counts=np.ones(256))
        self.Q.set_representation_levels(centroids)
        #os.system(f"cp -f {self.args.input} {DECODE_INPUT}_000.png") 
        #k = gray_image.read(f"{DECODE_INPUT}_", 0).astype(np.int16)
        k = io.imread(self.args.input).astype(np.int16)
        y = self.Q.decode(k)
        rate = gray_image.write(y, f"{gray_pixel_static_scalar_quantization.DECODE_OUTPUT}_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp {gray_pixel_static_scalar_quantization.DECODE_OUTPUT}_000.png {self.args.output}")
        logging.info(f"Witten {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        return rate

    def decode_(self):
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
        logging.info(f"Written {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        return rate

if __name__ == "__main__":
    logging.info(__doc__)
    logging.info(f"quantizer = {quantizer_name}")
    gray_pixel_static_scalar_quantization.parser.description = __doc__
    args = gray_pixel_static_scalar_quantization.parser.parse_known_args()[0]

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = Gray_Pixel_LloydMax_Quantization(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

