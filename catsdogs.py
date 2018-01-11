from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
import time

batch_size = 16
train_images_count = 20000
validate_images_count = 2000

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))

model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

image_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = image_datagen.flow_from_directory(
    'data/train',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = image_datagen.flow_from_directory(
    'data/validation',
    target_size=(150, 150),
    batch_size=batch_size,
    class_mode='categorical'
)

# 1 epochoj 500 (steps_per_epoch) kartu yra paimama po 16 (batch_size) fotkiu
# ir paleidziama per tinkla
model.fit_generator(
    train_generator,
    steps_per_epoch=train_images_count // batch_size,
    epochs=10,
    validation_data=validation_generator,
    validation_steps=validate_images_count // batch_size
)

print(validation_generator.class_indices)
model.save_weights('saved_weights.h5')
model.save('saved_model')