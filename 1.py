from __future__ import print_function

import numpy as np
import cv2
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, ZeroPadding2D, BatchNormalization,AveragePooling2D,Input,Add
from keras.layers import Conv2D, MaxPooling2D
from keras.initializers import glorot_uniform
from keras.models import Model, load_model
from keras.preprocessing import image
from keras.utils import layer_utils
from keras.applications.imagenet_utils import preprocess_input

num_class = 7
img_rows, img_cols = 48,48
batch_size = 32

train_data = r'C:\Users\RANVEER\Desktop\shiv\data sets\Emotion Classification\train'
validation_data = r'C:\Users\RANVEER\Desktop\shiv\data sets\Emotion Classification\validation'

train_data_generator = ImageDataGenerator(rescale=1./255, rotation_range= 30, shear_range= 0.3, zoom_range= 0.3, width_shift_range= 0.4, height_shift_range=0.3,horizontal_flip= True, vertical_flip= True)
validation_data_gen = ImageDataGenerator(rescale=1./255)

train_generator = train_data_generator.flow_from_directory(train_data, color_mode= 'grayscale', target_size= (img_rows,img_cols), batch_size= batch_size, class_mode= 'categorical', shuffle=True)
validation_generator = validation_data_gen.flow_from_directory(validation_data, color_mode= 'grayscale', target_size= (img_rows,img_cols), batch_size= batch_size, class_mode= 'categorical', shuffle=True)


model1 = Sequential()

#Block1
model1.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape = (img_rows, img_cols,1)))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer = 'he_normal', input_shape = (img_rows, img_cols,1)))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Dropout(0.2))

#Block2
model1.add(Conv2D(64, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Dropout(0.2))

#Block3
model1.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(Conv2D(128, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Dropout(0.2))

#Block4
model1.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(Conv2D(256, (3,3), padding = 'same', kernel_initializer = 'he_normal'))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(MaxPooling2D(pool_size=(2,2)))
model1.add(Dropout(0.2))

#Block5
model1.add(Flatten())
model1.add(Dense(64, kernel_initializer='he_normal'))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.5))

#Block6
model1.add(Dense(64, kernel_initializer='he_normal'))
model1.add(Activation('elu'))
model1.add(BatchNormalization())
model1.add(Dropout(0.5))


#Block
model1.add(Dense(num_class, kernel_initializer='he_normal'))
model1.add(Activation('softmax'))

model1.summary()

from keras.optimizers import  RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

checkpoint = ModelCheckpoint(r'C:\Users\RANVEER\Desktop\shiv\data sets\Emotion Classification\Emotion_little_vgg.h5',monitor='val_loss',mode = 'min', save_best_only=True, verbose=1)
earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=3, verbose=1, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=3, verbose=1, min_delta= 0.0001)
callbacks = [earlystop, checkpoint,reduce_lr]

model1.compile(loss='categorical_crossentropy', optimizer= Adam(lr=0.001), metrics=['accuracy'])

nb_trian_samples = 28821
nb_validation_samples = 7066
epochs = 30

results = model1.fit_generator(train_generator, steps_per_epoch=nb_trian_samples//batch_size, epochs=epochs, callbacks=callbacks, validation_data=validation_generator, validation_steps=nb_validation_samples//batch_size)

