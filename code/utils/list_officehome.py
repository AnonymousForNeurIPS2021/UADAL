import os
import random

## Unknown Classes in Target Domain
 #  In alphabetical order, the classes 11-20 are used as unknowns in the source domain and 21-31 as unknowns in the target domain, i.e., the unknown classes in the source and target domain are not shared
 # From Busto, P.P., Gall, J.: Open set domain adaptation. In: ICCV. (2017)

#class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork']
class_list = ['Alarm_Clock', 'Backpack', 'Batteries', 'Bed', 'Bike', 'Bottle', 'Bucket', 'Calculator', 'Calendar', 'Candles', 'Chair', 'Clipboards', 'Computer', 'Couch', 'Curtains', 'Desk_Lamp', 'Drill', 'Eraser', 'Exit_Sign', 'Fan', 'File_Cabinet', 'Flipflops', 'Flowers', 'Folder', 'Fork']#, 'unk']
whole_class_list = sorted(os.listdir('data/officehome/Art/images'))

assert len(class_list)==(25)
print(len(os.listdir('data/officehome/Art/images')))
print(len(os.listdir('data/officehome/Clipart/images')))
print(len(os.listdir('data/officehome/Product/images')))
print(len(os.listdir('data/officehome/RealWorld/images')))
unknown_list = list(set(whole_class_list)-set(class_list))
unknown_list.sort()
print('shared class :',len(class_list))
print('whole',len(whole_class_list))
print('unknowns',len(unknown_list))
print(unknown_list)
target_class_list = class_list + unknown_list
print(len(target_class_list))
data_path = os.path.join('data', 'officehome')
## office 11 classification
for domain in ['Art', 'Clipart', 'Product', 'RealWorld']:
    data_domain_path = os.path.join(data_path, domain)
    image_path = os.path.join(data_domain_path, 'images')
    dir_list = os.listdir(image_path)

    path_source_domain = os.path.join(data_domain_path, 'officehome_%s_source_list_v2.txt'%(domain))
    write_source_domain = open(path_source_domain, "w")
    path_target_domain = os.path.join(data_domain_path, 'officehome_%s_target_list_v2.txt'%(domain))
    write_target_domain = open(path_target_domain, "w")
    c0, c1, c2 = 0, 0, 0
    for k, direc in enumerate(dir_list):
        if not '.txt' in direc:
            files = os.listdir(os.path.join(image_path, direc))
            for i, file in enumerate(files):
                c0 +=1
                file_name = os.path.join(image_path, direc, file)
                if direc in class_list:
                    c1 +=1
                    class_name = direc
                    write_source_domain.write('%s %s\n' % (file_name, class_list.index(class_name)))
                    write_target_domain.write('%s %s\n' % (file_name, class_list.index(class_name)))
                else:
                    if direc in unknown_list:
                        c2+=1
                        class_name = direc#'unk'
                        write_target_domain.write('%s %s\n' % (file_name, target_class_list.index(class_name)))

    print(domain, c0, c1, c2)
