import numpy as np
import torch
import torch.utils.data as data
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from random import sample, random
import scipy.io as sio
import codecs
import os
import os.path


def _dataset_info(txt_labels, folder_dataset):
    with open(txt_labels, 'r') as f:
        images_list = f.readlines()

    file_names = []
    labels = []
    for row in images_list:
        row = row.split(' ')
        file_name = folder_dataset + row[0]
        file_names.append(file_name)
        labels.append(int(row[1]))

    return file_names, labels


def get_split_dataset_info(txt_list, folder_dataset):
    names, labels = _dataset_info(txt_list, folder_dataset)
    return names, labels


class CustomDataset(data.Dataset):
    def __init__(self, names, labels, img_transformer=None, returns=None, is_train=None, ss_classes=None,
                 n_classes=None, only_4_rotations=None, n_classes_target=None):
        self.data_path = ""
        self.names = names
        self.labels = labels
        self.N = len(self.names)
        self._image_transformer = img_transformer
        self.is_train = is_train
        self.returns = returns
        self.ss_classes = ss_classes
        self.n_classes = n_classes
        self.only_4_rotations = only_4_rotations
        self.n_classes_target = n_classes_target

    def __getitem__(self, index):
        #framename = self.data_path + '/' + self.names[index]
        framename = self.names[index]
        img = Image.open(framename).convert('RGB')

        if self.returns == 3:
            data, data_ss, label_ss = self._image_transformer(img, self.labels[index], self.is_train, self.ss_classes,
                                                              self.n_classes, self.only_4_rotations,
                                                              self.n_classes_target)
            return data, data_ss, label_ss
        elif self.returns == 4:
            data, data_ss, label, label_ss = self._image_transformer(img, self.labels[index], self.is_train,
                                                                     self.ss_classes, self.n_classes,
                                                                     self.only_4_rotations, self.n_classes_target)
            return data, data_ss, label, label_ss
        elif self.returns == 5:
            data, data_ss, label, label_ss, label_ss_center = self._image_transformer(img, self.labels[index],
                                                                                      self.is_train, self.ss_classes,
                                                                                      self.n_classes,
                                                                                      self.only_4_rotations,
                                                                                      self.n_classes_target)
            return data, data_ss, label, label_ss, label_ss_center
        elif self.returns == 6:
            data, data_ss, label, label_ss, label_ss_center, label_object_center = self._image_transformer(img,
                                                                                                           self.labels[
                                                                                                               index],
                                                                                                           self.is_train,
                                                                                                           self.ss_classes,
                                                                                                           self.n_classes,
                                                                                                           self.only_4_rotations,
                                                                                                           self.n_classes_target)
            return data, data_ss, label, label_ss, label_ss_center, label_object_center
        elif self.returns == 2:
            data, label = self._image_transformer(img, self.labels[index], self.is_train, self.ss_classes,
                                                  self.n_classes, self.only_4_rotations, self.n_classes_target)
            return data, label

    def __len__(self):
        return len(self.names)

import tensorlayer as tl
from skimage.transform import resize
from scipy.misc import imread, imresize
#import torchvision.transforms as transforms
def transform_source_ss(data, label, is_train, ss_classes, n_classes, only_4_rotations, n_classes_target):
    ss_transformation = np.random.randint(ss_classes)
    data = imresize(data, (256, 256))
    original_image = tl.prepro.crop(data, 224, 224, is_random=is_train)
    data = tl.prepro.crop(data, 224, 224, is_random=is_train)

    if ss_transformation == 0:
        ss_data = data
    if ss_transformation == 1:
        ss_data = np.rot90(data, k=1)
    if ss_transformation == 2:
        ss_data = np.rot90(data, k=2)
    if ss_transformation == 3:
        ss_data = np.rot90(data, k=3)

    if only_4_rotations:
        ss_label = one_hot(ss_classes, ss_transformation)
        label_ss_center = ss_transformation
    else:
        ss_label = one_hot(ss_classes * n_classes, (ss_classes * label) + ss_transformation)
        label_ss_center = (ss_classes * label) + ss_transformation

    ss_data = np.transpose(ss_data, [2, 0, 1])
    ss_data = np.asarray(ss_data, np.float32) / 255.0

    original_image = np.transpose(original_image, [2, 0, 1])
    original_image = np.asarray(original_image, np.float32) / 255.0
    label_object_center = label
    #label = one_hot(n_classes + 1, label)
    label = one_hot(n_classes, label)

    return original_image, ss_data, label, ss_label, label_ss_center, label_object_center


def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp