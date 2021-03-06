from keras.models import load_model
from keras.preprocessing import image
import os
import numpy as np

model = load_model('great_model_ducks')
path = "C:\\Dev\\AI\\2U\\data\\validation\\ducks"
totalImagesShown = 500
catProb = 0
dogProb = 0
duckProb = 0
for filename in (os.listdir(path)):
    img = image.load_img(os.path.join(path, filename), target_size=(150,150))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis = 0)
    pred = model.predict(x)
    catProb+=pred[0][0]
    dogProb+=pred[0][1]
    duckProb+=pred[0][2]

print ("Cat predictions: " + str(catProb/totalImagesShown))
print ("Dog predictions: " + str(dogProb/totalImagesShown))
print ("Duck predictions: " + str(duckProb/totalImagesShown))