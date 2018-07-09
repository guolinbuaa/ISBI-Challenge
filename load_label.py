from PIL import Image
from keras.preprocessing import image
import numpy as np

# def binarylab(labels,nb_class,input_shape):
#     y = np.zeros((input_shape[0], input_shape[1], nb_class))  # 224,224,21
#     for i in range(input_shape[0]):
#         for j in range(input_shape[1]):
#             y[i, j,int(labels[i][j])] = 1
#     return y

def load_label(label_path, input_shape, nb_class):
    y = Image.open(label_path)
    y = y.resize([input_shape[1], input_shape[0]])
    y.load()
    Y = image.img_to_array(y)
    Y = np.reshape(Y, [input_shape[0], input_shape[1]])
    y.close()

    mask = np.zeros((input_shape[0], input_shape[1], nb_class))

    for i in range(input_shape[0]):
        for j in range(input_shape[1]):
            if Y[i,j] == 255:
                mask[i, j, 1] = 1
            else:
                mask[i, j, 0] = 1

    mask = np.expand_dims(mask, axis=0)
    return mask
