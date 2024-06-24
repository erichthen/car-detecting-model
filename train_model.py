
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import shutil
import os

data = 'small_meta'
img_height, img_width = 224, 224

#augment the data, and split it into training and validation sets
data_prep = ImageDataGenerator(
    rescale = 1.0/255.0,
    shear_range = 0.2,
    zoom_range = 0.2,
    horizontal_flip = True,
    fill_mode = 'nearest',
    validation_split= 0.2)


#load the data from meta
train_gen = data_prep.flow_from_directory(
    data,
    target_size = (img_height, img_width),
    batch_size = 16,
    class_mode = 'categorical',
    subset = 'training')

validation_gen = data_prep.flow_from_directory(
    data,
    target_size = (img_height, img_width),
    batch_size = 16,
    class_mode = 'categorical',
    subset = 'validation')


#using efficientnet as the base model
base_model = EfficientNetB0(
    include_top = False,
    weights = 'imagenet',
    input_shape = (img_height, img_width, 3))


base_model = EfficientNetB0(weights='imagenet', include_top=False, input_shape=(img_height, img_width, 3))

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)
predictions = Dense(len(train_gen.class_indices), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)

for layer in base_model.layers:
    layer.trainable = False
    
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

history = model.fit(
    train_gen,
    validation_data=validation_gen,
    epochs=10,  
    verbose=1
)

model.save('model.h5')

