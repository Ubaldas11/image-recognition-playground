from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.models import Sequential, load_model

batch_size = 16
train_images_count = 25000
validate_images_count = 2500

model = load_model("great_model")
#prepare for 3 classes
model.summary()
model.pop()
model.add(Dense(3, activation='softmax', name="final_layer"))

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
model.save('great_model_ducks')

