from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense

image_datagen = ImageDataGenerator()

predict_generator = image_datagen.flow_from_directory(
    'data/predict',
    target_size=(150, 150),
    batch_size=1,
    class_mode=None,
    shuffle=False
)

model = load_model('saved_model')

prediction = model.predict_generator(predict_generator, steps=1)
print(prediction)