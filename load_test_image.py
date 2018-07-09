from PIL import Image
import numpy as np


def load_test_image(file_path, input_shape):
    tmp_img = Image.open(file_path)
    tmp_img = tmp_img.resize((input_shape[1], input_shape[0]))
    tmp_img.load()
    x = np.array(tmp_img)
    # x = np.reshape(x,(input_shape[0],input_shape[1],1))
    tmp_mean = np.mean(x)
    X = x-tmp_mean
    X /= 255.
    X = np.reshape(X, (input_shape[0], input_shape[1], 1))
    X = np.expand_dims(X, axis=0)
    tmp_img.close()

    return X
