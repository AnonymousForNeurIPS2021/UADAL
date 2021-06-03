import torch.utils.data as data
from PIL import Image
import os
import os.path
import numpy as np
import torch

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]


def find_classes(dir):
    classes = [d for d in os.listdir(dir) if os.path.isdir(os.path.join(dir, d))]
    classes.sort()
    class_to_idx = {classes[i]: i for i in range(len(classes))}
    return classes, class_to_idx


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir, class_to_idx):
    images = []
    dir = os.path.expanduser(dir)
    for target in os.listdir(dir):
        d = os.path.join(dir, target)
        if not os.path.isdir(d):
            continue

        for root, _, fnames in sorted(os.walk(d)):
            for fname in fnames:
                if is_image_file(fname):
                    path = os.path.join(root, fname)
                    item = (path, class_to_idx[target])
                    images.append(item)

    return images


def default_flist_reader(flist):
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


def default_loader(path):
    return Image.open(path).convert('RGB')


def make_dataset_nolist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
        # print(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


class ImageFolder(data.Dataset):
    def __init__(self, image_list, transform=None, target_transform=None, return_paths=False,
                 loader=default_loader,train=False):
        imgs, labels = make_dataset_nolist(image_list)
        #self.root = root
        self.imgs = imgs
        self.labels= labels
        #self.classes = classes
        #self.class_to_idx = class_to_idx
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths
        self.train = train
    def __getitem__(self, index):
        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)
        #if self.train:
        #    img = augment_images(img)

        img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return img, target, path
        else:
            return img, target

    def __len__(self):
        return len(self.imgs)


def one_hot(n_class, index):
    tmp = np.zeros((n_class,), dtype=np.float32)
    tmp[index] = 1.0
    return tmp

class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None, flist_reader=default_flist_reader,
                 loader=default_loader, return_paths=True):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        impath = impath.replace('other','unk')
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return img, target, impath
        else:
            return img, target

    def __len__(self):
        return len(self.imlist)


