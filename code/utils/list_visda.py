import os
import random

import os
import random
data_path = os.path.join('data', 'visda')

p_path = os.path.join(data_path, 'train')
dir_list = os.listdir(p_path)
path_source = os.path.join(p_path, 'source_list_v2.txt') #"utils/source_list.txt"
write_source = open(path_source,"w")
print(dir_list)

class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck", "unk"]

visda_target_class_list = ["bicycle", "bus", "car", "motorcycle", "train", "truck"] + ['aeroplane', 'horse', 'knife', 'person', 'plant', 'skateboard']
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in class_list:
                class_name = direc
                file_name = os.path.join(p_path, direc, file)
                write_source.write('%s %s\n' % (file_name, class_list.index(class_name)))
            else:
                continue

p_path = os.path.join(data_path, 'validation')
dir_list = os.listdir(p_path)
path_target = os.path.join(p_path, 'target_list_v2.txt')
write_target = open(path_target, "w")

print(dir_list)
for k, direc in enumerate(dir_list):
    if not '.txt' in direc:
        files = os.listdir(os.path.join(p_path, direc))
        for i, file in enumerate(files):
            if direc in visda_target_class_list:
                class_name = direc
                file_name = os.path.join(p_path, direc, file)
                write_target.write('%s %s\n' % (file_name, visda_target_class_list.index(class_name)))





