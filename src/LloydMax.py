'''Image quantization using a LloydMax quantizer.'''

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

import entropy

entropy.parser.add_argument("-q", "--QSS", type=entropy.int_or_str, help=f"Quantization step size (default: 32)", default=32)

class LloydMax_Quantizer(entropy.Entropy_Codec):
    
    def __init__(self, args): # ??
        super().__init__(args)

    def encode(self):
        '''Read an image, quantize the image, and save it.'''
        img = self.read()
        with open(f"{self.args.output}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")
        rate = 1*8/(img.shape[0]*img.shape[1]) # We suppose that the representation of the QSS requires 1 byte
        logging.info(f"Written {self.args.output}_QSS.txt")
        #extended_img = np.expand_dims(img, axis=2)
        #k = np.empty_like(extended_img)
        k = np.empty_like(img)
        print(k.shape)
        #for c in range(extended_img.shape[2]):
        for c in range(img.shape[2]):
            #histogram_img, bin_edges_img = np.histogramdd(extended_img[..., c], bins=256, range=(0, 256))
            histogram_img, bin_edges_img = np.histogram(img[..., c], bins=256, range=(0, 256))
            logging.info(f"histogram = {histogram_img}")
            self.Q = Quantizer(Q_step=self.args.QSS, counts=histogram_img)
            centroids = self.Q.get_representation_levels()
            with gzip.GzipFile(f"{self.args.output}_centroids_{c}.gz", "w") as f:
                np.save(file=f, arr=centroids)
            len_codebook = os.path.getsize(f"{self.args.output}_centroids_{c}.gz")
            logging.info(f"Written {len_codebook} bytes in {self.args.output}_centroids_{c}.gz")
            rate += len_codebook/8/(img.shape[0]*img.shape[1])
            #k[..., c] = self.Q.encode(extended_img[..., c])
            k[..., c] = self.Q.encode(img[..., c])
            k[..., c] = k[..., c].astype(np.uint8)
        rate += gray_image.write(k, f"{entropy.ENCODE_OUTPUT}_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp {entropy.ENCODE_OUTPUT}_000.png {self.args.output}")
        logging.info(f"Written {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        return rate

    def decode(self):
        with open(f"{self.args.input}_QSS.txt", 'r') as f:
            QSS = int(f.read())
        logging.info(f"Read QSS={QSS} from {self.args.output}_QSS.txt")
        with gzip.GzipFile(f"{self.args.input}_centroids.gz", "r") as f:
            centroids = np.load(file=f)
        logging.info(f"Read {self.args.input}_centroids.gz")
        self.Q = Quantizer(Q_step=QSS, counts=np.ones(shape=256))
        self.Q.set_representation_levels(centroids)
        #os.system(f"cp -f {self.args.input} {DECODE_INPUT}_000.png") 
        #k = gray_image.read(f"{DECODE_INPUT}_", 0).astype(np.int16)
        k = io.imread(self.args.input).astype(np.int16)
        y = self.Q.decode(k)
        rate = gray_image.write(y, f"{entropy.DECODE_OUTPUT}_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp {entropy.DECODE_OUTPUT}_000.png {self.args.output}")
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
    entropy.parser.description = __doc__
    args = entropy.parser.parse_known_args()[0]

    try:
        logging.info(f"input = {args.input}")
        logging.info(f"output = {args.output}")
    except AttributeError:
        logging.error("You must specify 'encode' or 'decode'")
        quit()

    codec = LloydMax_Quantizer(args)

    rate = args.func(codec)
    logging.info(f"rate = {rate} bits/pixel")

