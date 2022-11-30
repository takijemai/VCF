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
        logging.info(f"QSS = {self.args.QSS}")
        with open(f"{self.args.output}_QSS.txt", 'w') as f:
            f.write(f"{self.args.QSS}")
        rate = 1*8/(img.shape[0]*img.shape[1]) # We suppose that the representation of the QSS requires 1 byte
        logging.info(f"Written {self.args.output}_QSS.txt")
        if len(img.shape) < 3:
            extended_img = np.expand_dims(img, axis=2)
        else:
            extended_img = img
        k = np.empty_like(extended_img)
        #k = np.empty_like(img)
        print(extended_img.shape)
        for c in range(extended_img.shape[2]):
        #for c in range(img.shape[2]):
            histogram_img, bin_edges_img = np.histogram(extended_img[..., c], bins=256, range=(0, 256))
            #histogram_img, bin_edges_img = np.histogram(img[..., c], bins=256, range=(0, 256))
            logging.info(f"histogram = {histogram_img}")
            self.Q = Quantizer(Q_step=self.args.QSS, counts=histogram_img)
            centroids = self.Q.get_representation_levels()
            with gzip.GzipFile(f"{self.args.output}_centroids_{c}.gz", "w") as f:
                np.save(file=f, arr=centroids)
            len_codebook = os.path.getsize(f"{self.args.output}_centroids_{c}.gz")
            logging.info(f"Written {len_codebook} bytes in {self.args.output}_centroids_{c}.gz")
            rate += len_codebook/8/(img.shape[0]*img.shape[1])
            k[..., c] = self.Q.encode(extended_img[..., c])
            #k[..., c] = self.Q.encode(img[..., c])
            k[..., c] = k[..., c].astype(np.uint8)
        print(k.shape)
        rate += gray_image.write(k, f"{entropy.ENCODE_OUTPUT}_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp {entropy.ENCODE_OUTPUT}_000.png {self.args.output}")
        logging.info(f"Written {os.path.getsize(self.args.output)} bytes in {self.args.output}")
        return rate

    def decode(self):
        print(self.args.input)
        k = io.imread(self.args.input)
        print(k.shape)
        k = k.astype(np.int16)
        with open(f"{self.args.input}_QSS.txt", 'r') as f:
            QSS = int(f.read())
        logging.info(f"Read QSS={QSS} from {self.args.output}_QSS.txt")
        if len(k.shape) < 3:
            extended_k = np.expand_dims(k, axis=2)
        else:
            extended_k = k
        print(k.shape)
        y = np.empty_like(extended_k)
        for c in range(y.shape[2]):
            with gzip.GzipFile(f"{self.args.input}_centroids_{c}.gz", "r") as f:
                centroids = np.load(file=f)
            logging.info(f"Read {self.args.input}_centroids_{c}.gz")
            self.Q = Quantizer(Q_step=QSS, counts=np.ones(shape=256))
            self.Q.set_representation_levels(centroids)
            y[..., c] = self.Q.decode(extended_k[..., c])
        rate = gray_image.write(y, f"{entropy.DECODE_OUTPUT}_", 0)*8/(k.shape[0]*k.shape[1])
        os.system(f"cp {entropy.DECODE_OUTPUT}_000.png {self.args.output}")
        logging.info(f"Witten {os.path.getsize(self.args.output)} bytes in {self.args.output}")
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

