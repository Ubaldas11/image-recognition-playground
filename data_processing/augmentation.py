from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
import os

datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

path = "C:\\Dev\\AI\\2U\\data\\train\\ducks"
aug_value = 10
j = 0
for idx, filename in enumerate(os.listdir(path)):
    j+=1
    if j == 60:
        break
    img = load_img(os.path.join(path, filename))
    x = img_to_array(img)
    x = x.reshape((1,) + x.shape)
    print ("Augmenting image " + filename + " " + str(aug_value) + " times.")
    i = 0
    for batch in datagen.flow(x, batch_size=1,
                          save_to_dir='data/augmentedDucksTrain',
                          save_prefix='duck',
                          save_format='jpg'):
        i += 1
        if (i > aug_value):
            break