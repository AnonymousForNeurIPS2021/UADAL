import os
import random
class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork']#, 'unk']
whole_class_list = sorted(os.listdir('data/officehome/Art/images'))

assert len(class_list)==(25)
unknown_list = list(set(whole_class_list)-set(class_list))
unknown_list.sort()

target_class_list = class_list + unknown_list

data_path = os.path.join('data', 'officehome')

for domain in ['Art', 'Clipart', 'Product', 'RealWorld']:
    data_domain_path = os.path.join(data_path, domain)
    image_path = os.path.join(data_domain_path, 'images')
    dir_list = os.listdir(image_path)

    path_source_domain = os.path.join(data_domain_path, 'officehome_%s_source_list_v2.txt'%(domain))
    write_source_domain = open(path_source_domain, "w")
    path_target_domain = os.path.join(data_domain_path, 'officehome_%s_target_list_v2.txt'%(domain))
    write_target_domain = open(path_target_domain, "w")

    for k, direc in enumerate(dir_list):
        if not '.txt' in direc:
            files = os.listdir(os.path.join(image_path, direc))
            for i, file in enumerate(files):
                file_name = os.path.join(image_path, direc, file)
                if direc in class_list:
                    class_name = direc
                    write_source_domain.write('%s %s\n' % (file_name, class_list.index(class_name)))
                    write_target_domain.write('%s %s\n' % (file_name, class_list.index(class_name)))
                else:
                    if direc in unknown_list:
                        class_name = direc
                        write_target_domain.write('%s %s\n' % (file_name, target_class_list.index(class_name)))


