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



class ImageFolder_ss(data.Dataset):
    def __init__(self, image_list, transform, ss_classes, known_classes, target_transform=None, return_paths=False,
                 loader=default_loader,train=False):

        imgs, labels = make_dataset_nolist(image_list)
        self.imgs = imgs
        self.labels= labels
        self.ss_classes = ss_classes
        self.known_classes = known_classes
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader
        self.return_paths = return_paths
        self.train = train
    def __getitem__(self, index):
        ss_transformation = np.random.randint(self.ss_classes)

        path = self.imgs[index]
        target = self.labels[index]
        img = self.loader(path)

        img = self.transform(img).numpy()
        img = np.transpose(img, [1, 2, 0])
            #print(type(img))
        if ss_transformation == 0:
            ss_data = img
        if ss_transformation == 1:
            ss_data = np.rot90(img, k=1)

        if ss_transformation == 2:
            ss_data = np.rot90(img, k=2)

        if ss_transformation == 3:
            ss_data = np.rot90(img, k=3)

        img = np.transpose(img, [2, 0, 1])
        ss_data = np.transpose(ss_data, [2, 0, 1])

        img = torch.from_numpy(img.copy())
        ss_data = torch.from_numpy(ss_data.copy())

        if self.known_classes == 1:
            ss_label = one_hot(self.ss_classes, ss_transformation)
            label_ss_center = 0
            label_object_center = 0
        else:
            ss_label = one_hot(self.ss_classes * self.known_classes, (self.ss_classes * target) + ss_transformation)
            label_ss_center = (self.ss_classes * target) + ss_transformation
            label_object_center = target


        if self.target_transform is not None:
            target = self.target_transform(target)
        if self.return_paths:
            return img, target, path, ss_data, ss_label, label_ss_center, label_object_center
        else:
            return img, target, ss_data, ss_label, label_ss_center, label_object_center

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


