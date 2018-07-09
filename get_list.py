import os
import numpy as np

def get_list(file_path):

    image_list = []
    label_list = []
    files = os.listdir(file_path)

    for file in files:
        if 'mask' in file:
            continue
        else:
            temp_image_path = os.path.join(file_path, file)
            image_list.append(temp_image_path)
            temp_split_file = file.split('.')
            temp_label_file = temp_split_file[0] + '_mask' + '.tif'
            if temp_label_file in files:
                temp_label_path = os.path.join(file_path, temp_label_file)
                label_list.append(temp_label_path)

    return image_list, label_list

