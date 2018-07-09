import numpy as np
from get_list import get_list
from myGenerator import mygen
# from segnet import SegNet
from keras.callbacks import TensorBoard, ModelCheckpoint, ReduceLROnPlateau
from keras.utils import plot_model
from unet_model import get_unet

batch_size = 12
n_classes = 2
EPOCH = 100
img_W = 160
img_H = 240
input_shape = (img_W, img_H)

train_file_path = 'raw/train/'
test_file_path = 'raw/test/'

train_image_list, train_label_list = get_list(file_path=train_file_path)
test_image_list, test_label_list = get_list(file_path=test_file_path)

train_list = np.vstack((train_image_list, train_label_list))
train_list = np.transpose(train_list)

# test_list = np.vstack((test_image_list, test_label_list))
# test_list = np.transpose(test_list)

train_data_generator = mygen(train_list, input_shape, n_classes, batch_size)

val_data_generator = mygen(train_list, input_shape, n_classes, batch_size)

# model = SegNet(input_shape, n_classes)
model = get_unet(input_shape=(input_shape[0],input_shape[1],1), n_class=n_classes)

plot_model(model, to_file='unet_model.png', show_shapes=1)

model.compile(optimizer='Adam', loss='binary_crossentropy', metrics=['accuracy'])

steps_per_epoch = len(train_list)//batch_size

TB = TensorBoard(log_dir='log')
MC = ModelCheckpoint(filepath='unet_best_model.h5',save_best_only=1)
LR = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=10,
                       verbose=0, mode='auto', epsilon=0.0001, cooldown=0, min_lr=0)
CB = [TB, MC, LR]

model.fit_generator(train_data_generator, epochs=EPOCH,
                    steps_per_epoch=steps_per_epoch, verbose=1, callbacks=CB,
                    validation_data=val_data_generator, validation_steps=1)

# print('')