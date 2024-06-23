

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.models import Model

import os

data_dir = 'meta'
img_height, img_width = 224, 224

data_prep = ImageDataGenerator(
    rescale = 1.0/255.0,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest',
    validation_split= 0.2
)

train_gen = data_prep.flow_from_directory(
    data_dir,
    target_size = (img_height, img_width),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'training'
)

validation_gen = data_prep.flow_from_directory(
    data_prep,
    target_size = (img_height, img_width),
    batch_size = 32,
    class_mode = 'categorical',
    subset = 'validation',
)

base_model = EfficientNetB0(
include_top=False, weights='imagenet',
input_shape=(img_height, img_width, 3)
)



model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dense(1024, activation='relu'),
    Dense(len(train_gen.class_indices), activation='softmax')
])

model.compile(optimizer = Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])


    
#print("Class indices:", train_gen.class_indices)
#print("Class indices:", validation_gen.class_indices)

