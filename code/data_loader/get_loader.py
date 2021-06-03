from .mydataset import ImageFolder, ImageFilelist
from .unaligned_data_loader import UnalignedDataLoader
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import sys
import numpy as np
from collections import Counter


def get_loader(source_path, target_path, evaluation_path, transforms, batch_size=32):
    sampler = None

    source_folder = ImageFolder(os.path.join(source_path),
                                transforms[source_path])
    target_folder_train = ImageFolder(os.path.join(target_path),
                                         transforms[source_path])
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                     transform=transforms[evaluation_path],
                                     return_paths=True)
    source_ss_folder = ImageFolder(os.path.join(source_path), transforms[source_path], return_paths=False)

    freq = Counter(source_folder.labels)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in source_folder.labels]
    sampler = WeightedRandomSampler(source_weights,
                                    len(source_folder.labels))

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size, sampler=sampler)

    pin = True
    num_workers = 2
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)

    if sampler is not None:
        train_loader_ss = torch.utils.data.DataLoader(source_ss_folder, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin, drop_last=True)
    else:
        train_loader_ss = torch.utils.data.DataLoader(source_ss_folder, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True)

    return train_loader, test_loader, train_loader_ss



def get_dataset_information(dataset, s_d, t_d):
    if dataset == 'office31':
        name_dict = {'A':'amazon', 'D':'dslr', 'W':'webcam'}
        data_path = os.path.join('data', 'office')
        source_path = os.path.join(os.path.join(data_path, name_dict[s_d]), 'office31_%s_source_list_v2.txt'%name_dict[s_d])
        target_path = os.path.join(os.path.join(data_path, name_dict[t_d]), 'office31_%s_target_list_v2.txt'%name_dict[t_d])
        evaluation_data = os.path.join(os.path.join(data_path, name_dict[t_d]), 'office31_%s_target_list_v2.txt'%name_dict[t_d])
        class_list = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', "unk"]
        num_class = len(class_list) #11
    elif dataset == 'officehome':
        name_dict = {'A': 'Art', 'C': 'Clipart', 'P': 'Product', 'R':'RealWorld'}
        data_path = os.path.join('data', 'officehome')
        source_path = os.path.join(os.path.join(data_path, name_dict[s_d]), 'officehome_%s_source_list_v2.txt'%name_dict[s_d])
        target_path = os.path.join(os.path.join(data_path, name_dict[t_d]), 'officehome_%s_target_list_v2.txt'%name_dict[t_d])
        evaluation_data = os.path.join(os.path.join(data_path, name_dict[t_d]), 'officehome_%s_target_list_v2.txt'%name_dict[t_d])
        class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator',
                      'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp',
                      'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork',
                      'unk']
        num_class = len(class_list) #26
    elif dataset =='visda':
        s_d, t_d = 'train', 'validation'
        data_path = os.path.join('data', dataset)
        source_path = os.path.join(os.path.join(data_path, s_d), 'source_list_v2.txt')
        target_path = os.path.join(os.path.join(data_path, t_d), 'target_list_v2.txt')
        evaluation_data = os.path.join(os.path.join(data_path, t_d), 'target_list_v2.txt')
        class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]
        num_class = len(class_list) #7
    else:
        print('Specify the name of dataset!!')
        sys.exit()

    return source_path, target_path, evaluation_data, num_class

