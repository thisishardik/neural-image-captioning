import tensorflow as tf

import matplotlib.pyplot as plt

import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image


class Dataset:
    def __init__(self, annotation_file=None, PATH=None):
        self.annotation_file = annotation_file
        self.PATH = PATH

    def get_data(self):
        annotation_folder = "/annotations/"
        if not os.path.exists(os.path.abspath(".") + annotation_folder):
            annotation_zip = tf.keras.utils.get_file(
                "captions.zip",
                cache_subdir=os.path.abspath("."),
                origin="http://images.cocodataset.org/annotations/annotations_trainval2014.zip",
                extract=True,
            )
            self.annotation_file = (
                os.path.dirname(annotation_zip) + "/annotations/captions_train2014.json"
            )
            os.remove(annotation_zip)

        image_folder = "/train2014/"
        if not os.path.exists(os.path.abspath(".") + image_folder):
            image_zip = tf.keras.utils.get_file(
                "train2014.zip",
                cache_subdir=os.path.abspath("."),
                origin="http://images.cocodataset.org/zips/train2014.zip",
                extract=True,
            )
            self.PATH = os.path.dirname(image_zip) + image_folder
            os.remove(image_zip)
        else:
            self.PATH = os.path.abspath(".") + image_folder

    def grouping(self):
        with open(self.annotation_file, "r") as f:
            annotations = json.load(f)


if __name__ == "__main__":
    get_data()
