from .mydataset import ImageFolder, ImageFilelist, ImageFolder_ss
from .unaligned_data_loader import UnalignedDataLoader
import os
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import sys
import numpy as np
# seed =1
# torch.manual_seed(seed)
# torch.cuda.manual_seed(seed)
# np.random.seed(1)
from collections import Counter


def get_loader_ss(args, source_path, target_path, evaluation_path, transforms, batch_size=32, ss_classes=4, known_classes=25):
    sampler = None

    source_folder = ImageFolder_ss(os.path.join(source_path),
                                transforms[source_path], ss_classes, known_classes)
    target_folder_train = ImageFolder_ss(os.path.join(target_path),
                                         transforms[source_path], ss_classes, 1)
    eval_folder_test = ImageFolder(os.path.join(evaluation_path),
                                     transform=transforms[evaluation_path],
                                     return_paths=True)
    eval_folder_test_source = ImageFolder(os.path.join(source_path),
                                     transform=transforms[evaluation_path],
                                     return_paths=True)
    source_ss_folder = ImageFolder_ss(os.path.join(source_path), transforms[source_path], ss_classes, known_classes, return_paths=False)
    target_ss_folder = ImageFolder_ss(os.path.join(target_path), transforms[target_path], ss_classes, 1, return_paths=False)

    freq = Counter(source_folder.labels)
    class_weight = {x: 1.0 / freq[x] for x in freq}
    source_weights = [class_weight[x] for x in source_folder.labels]
    sampler = WeightedRandomSampler(source_weights,
                                    len(source_folder.labels))

    train_loader = UnalignedDataLoader()
    train_loader.initialize(source_folder, target_folder_train, batch_size, sampler=sampler, ss_signal=True)

    pin = True
    num_workers = 2
    test_loader = torch.utils.data.DataLoader(
        eval_folder_test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)

    test_loader_source = torch.utils.data.DataLoader(
        eval_folder_test_source,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers, pin_memory=pin)

    if sampler is not None:
        train_loader_ss = torch.utils.data.DataLoader(source_ss_folder, batch_size=batch_size, sampler=sampler, num_workers=num_workers, pin_memory=pin, drop_last=True)
    else:
        train_loader_ss = torch.utils.data.DataLoader(source_ss_folder, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin, drop_last=True)
    train_loader_st = torch.utils.data.DataLoader(target_ss_folder, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)

    return train_loader, test_loader, test_loader_source, train_loader_ss, train_loader_st



def get_dataset_information(dataset, s_d, t_d):
    len_dict11 = {'A':{'s':824 , 't':1836 }, 'D':{'s':154 , 't':342}, 'W':{'s':235 , 't':502}}
    len_dicthome = {'A': {'s': 1089, 't': 2427}, 'C': {'s': 1675, 't': 4365}, 'P': {'s': 1785, 't': 4439}, 'R': {'s': 1811, 't': 4357}}
    if dataset == 'office31':
        name_dict = {'A':'amazon', 'D':'dslr', 'W':'webcam'}
        data_path = os.path.join('data', 'office')
        source_path = os.path.join(os.path.join(data_path, name_dict[s_d]), 'office11_%s_source_list_v2.txt'%name_dict[s_d])
        target_path = os.path.join(os.path.join(data_path, name_dict[t_d]), 'office11_%s_target_list_v2.txt'%name_dict[t_d])
        evaluation_data = os.path.join(os.path.join(data_path, name_dict[t_d]), 'office11_%s_target_list_v2.txt'%name_dict[t_d])
        class_list = ['back_pack', 'bike', 'bike_helmet', 'bookcase', 'bottle', 'calculator', 'desk_chair', 'desk_lamp', 'desktop_computer', 'file_cabinet', "unk"]
        num_class = len(class_list) #11
        len_s, len_t = len_dict11[s_d]['s'], len_dict11[t_d]['t']
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
        len_s, len_t = len_dicthome[s_d]['s'], len_dicthome[t_d]['t']
    elif dataset =='visda':
        s_d, t_d = 'train', 'validation'
        data_path = os.path.join('data', dataset)
        source_path = os.path.join(os.path.join(data_path, s_d), 'source_list_v2.txt')
        target_path = os.path.join(os.path.join(data_path, t_d), 'target_list_v2.txt')
        evaluation_data = os.path.join(os.path.join(data_path, t_d), 'target_list_v2.txt')
        class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]
        num_class = len(class_list) #7
        len_s, len_t = 79765, 55388
    else:
        print('Specify the name of dataset!!')
        sys.exit()

    return source_path, target_path, evaluation_data, num_class, class_list, len_s, len_t

