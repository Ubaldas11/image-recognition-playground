import os

path = "C:\\Dev\\AI\\2U\\data\\validation\\cats"
for idx, filename in enumerate(os.listdir(path)):
    os.rename(os.path.join(path, filename), (os.path.join(path, str(idx + 1000) + ".jpg")))
    print(filename)