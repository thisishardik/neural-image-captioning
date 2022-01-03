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
    def __init__(self, annotation_file=None, annotations=None, PATH=None):
        self.annotation_file = annotation_file
        self.PATH = PATH
        self.annotations = annotations

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

    def group_data(self):
        self.get_data()
        with open(self.annotation_file, "r") as f:
            self.annotations = json.load(f)
        image_path_to_caption = collections.defaultdict(list)
        for val in self.annotations["annotations"]:
            caption = f"<start> {val['caption']} <end>"
            image_path = self.PATH + "COCO_train2014_" + "%012d.jpg" % (val["image_id"])
            image_path_to_caption[image_path].append(caption)
        image_paths = list(image_path_to_caption.keys())
        random.shuffle(image_paths)
        train_image_paths = image_paths[:6000]
        print(len("Training samples:", train_image_paths))
        return image_path_to_caption, train_image_paths

    def fetch_data(self):
        image_path_to_caption, train_image_paths = self.group_data()
        train_captions = []
        img_name_vector = []

        for image_path in train_image_paths:
            caption_list = image_path_to_caption[image_path]
            train_captions.extend(caption_list)
            img_name_vector.extend([image_path] * len(caption_list))

        return train_captions, img_name_vector


if __name__ == "__main__":
    dataset = Dataset()
    train_captions, img_name_vector = dataset.fetch_data()
    print(train_captions[0])
    Image.open(img_name_vector[0])
