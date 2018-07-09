import numpy as np
from load_image import load_image
from load_label import load_label


def mygen(file_list, input_shape, nb_class, batch_size):
    while 1:
        data = np.zeros((1, input_shape[0], input_shape[1], 1))
        label = np.zeros((1, input_shape[0], input_shape[1], nb_class))

        num_file = len(file_list)

        idx = np.random.randint(low=0, high=num_file-1, size=batch_size)

        for temp_idx in idx:
            temp_image_list = file_list[temp_idx, 0]
            temp_image = load_image(temp_image_list, input_shape)
            data = np.vstack((data, temp_image))

            temp_label_list = file_list[temp_idx, 1]
            temp_label = load_label(temp_label_list, input_shape, nb_class)
            label = np.vstack((label, temp_label))

        data = data[1:, :, :, :]
        label = label[1:, :, :, :]
        train_data_generator = [data, label]
        yield train_data_generator

