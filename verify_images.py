import glob
from tensorflow.io import read_file, decode_image
from tensorflow.python.framework.errors_impl import InvalidArgumentError
import os

DATASET_PATH = os.environ["DATASET_PATH"]

# ensure that all images are readable by tensorflow
# delete any non-readable images

if __name__ == "__main__":
    images = glob.glob(DATASET_PATH + "*/*.png")
    for image in images:
        try:
            f = read_file(image)
            decode_image(f)
        except InvalidArgumentError:
            print(image)
            os.remove(image)
