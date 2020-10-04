from __future__ import print_function

import numpy as np
import cv2
import os
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Activation, Dropout, Flatten, Dense, Conv2D, MaxPool2D, ZeroPadding2D, BatchNormalization,AveragePooling2D,Input,Add
from keras.layers import Conv2D, MaxPool2D
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


def identity_layer(X,f,filters):

  #Retrieving Filters F1-64,F2-64,F3-256
  F1,F2,F3 = filters

  X_shortcut = X

  #First Layer
  X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid')(X) #(X) incicating input of previous layer to be output of this layer
  X = BatchNormalization(axis = 3)(X) # Batchnormalisation is done ater every layer in identity block. again (X) is output from this layer to next, axis 3 is for 3 color channels
  X = Activation('relu')(X)

  #Second Layer (f,f = 3,3) by default
  X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same')(X) #second layer has padding as same in ResNet
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)

  #Third Layer
  X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid')(X)
  X = BatchNormalization(axis = 3)(X)
  #no activation after this layer

  #Final step is to add the original X to convolved X after 3 layers
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)
  #final activation in this stage. Adding X_shortcut here to get original X value

  return X

def convolution_layer(X,f,filters, s=2):

  #Retrieving Filters F1-64,F2-64,F3-256
  F1,F2,F3 = filters

  X_shortcut = X

  #First Layer
  X = Conv2D(filters = F1, kernel_size = (1,1), strides = (s,s))(X) #First layer wil have stride 2. padding will be taken up by default
  X = BatchNormalization(axis = 3)(X) # Batchnormalisation is done ater every layer in identity block. again (X) is output from this layer to next, axis 3 is for 3 color channels
  X = Activation('relu')(X)

  #Second Layer (f,f = 3,3) by default
  X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same')(X) #second layer has padding as same in ResNet
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)

  #Third Layer
  X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid')(X)
  X = BatchNormalization(axis = 3)(X)
  #no activation after this layer

  #Shortcut path of the 1,1 convolution layer
  X_shortcut = Conv2D(filters = F3, kernel_size = (1,1), strides = (s,s), padding = 'valid')(X_shortcut)
  X_shortcut = BatchNormalization(axis = 3)(X_shortcut)
  #no activation here as well

  #Final step is to add the original X to convolved X after 3 layers
  X = Add()([X, X_shortcut])
  X = Activation('relu')(X)
  #final activation in this stage. Adding X_shortcut here to get original X value

  return X

from keras.layers import MaxPooling2D


def ResNet50(input_shape = (48,48,1), classes = 7):#shape and classes can vary based on our requirement




  #Define input with input_shape
  X_input = Input(input_shape) #accepts the first input as it is

  #Zero-Padding
  X = ZeroPadding2D((3,3))(X_input) #applying zero padding on the input

  #1st Stage - very 1st input to the model
  X = Conv2D(filters = 64, kernel_size = (7,7), strides = (2,2))(X)
  X = BatchNormalization(axis = 3)(X)
  X = Activation('relu')(X)
  X = MaxPooling2D((3,3), strides = (2,2))(X)


  #Stage 2 - here we will call a convolution layer
  X = convolution_layer(X, 3, [64,64,256], s=1) #Responsible for leveling input and output volume - shortcut to appy 3 layers defined above, the def func will take up the values from this func
  #instead of calling the function, we can also write the three layers as
  #X = Conv2D(F1, (1,1), strides = (s,s))(X)
  #X = Conv2D(F2, (f,f), strides = (1,1), padding = 'same')(X)
  #X = Conv2D(F3, (1,1), strides = (s,s),name = conv_name_base +'2a')(X)

  #Now we will use an identity block since output from conv layer wil be the same as output from below identify layer
  X = identity_layer(X, f =3, filters = [64,64,256]) #the identify_layer func will use these values for filters
  #instead of calling the function, we can also write the three layers as
  #X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid')(X)
  #X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same')(X)
  #X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid')(X)

  #Now we will use an identity block again since output from conv layer wil be the same as output from below identify layer
  X = identity_layer(X, 3, [64,64,256]) #the identify_layer func will use these values for filters
  #instead of calling the function, we can also write the three layers as
  #X = Conv2D(filters = F1, kernel_size = (1,1), strides = (1,1), padding = 'valid')(X)
  #X = Conv2D(filters = F2, kernel_size = (f,f), strides = (1,1), padding = 'same')(X)
  #X = Conv2D(filters = F3, kernel_size = (1,1), strides = (1,1), padding = 'valid')(X)

  #Stage 3 - Now we will use a convolution layer with 3 identity layers
  X = convolution_layer(X, f=3, filters = [128,128,512], s=2)
  X = identity_layer(X, 3, [128,128,512])
  X = identity_layer(X, 3, [128,128,512])
  X = identity_layer(X, 3, [128,128,512])

  #Stage 4 will have one conv layer with 5 identity layers
  X = convolution_layer(X, f=3, filters = [256,256,1024], s=2)
  X = identity_layer(X, 3, [256,256,1024])
  X = identity_layer(X, 3, [256,256,1024])
  X = identity_layer(X, 3, [256,256,1024])
  X = identity_layer(X, 3, [256,256,1024])
  X = identity_layer(X, 3, [256,256,1024])

  #Stage 5 will have one conv layer with 2 identity layers
  X = convolution_layer(X, f=3, filters = [512,512,2048], s=2)
  X = identity_layer(X, 3, [512,512,2048])
  X = identity_layer(X, 3, [512,512,2048])

  #Now will use Average pooling
  X = AveragePooling2D((2,2), name = 'avg_pool')(X)

  ##Code ends here##

  #Output layer
  X = Flatten()(X)
  X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer= glorot_uniform(seed = 0))(X)

  #Create Model
  model = Model(inputs = X_input, outputs = X, name = 'ResNet50')

  return model

model = ResNet50(input_shape = (48,48,1), classes = 7)
model.compile(optimizer = 'Adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

from tensorflow.keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='val_loss', patience= 2)
model.summary()
model.fit(train_generator, epochs= 15, validation_data= validation_generator, batch_size = 32, verbose = 1)
