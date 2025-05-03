import os
import numpy as np
from keras.models import load_model
import cv2
import config
import glob
import time

my_model = load_model(os.path.join("models","VGG16-f-w-max.keras"))

import pickle
file = open('le.pkl', 'rb')
le = pickle.load(file)
file.close()

raw_folder = "data/validate"
time__ = 0

for i_class in config.class_names:
    read_path = os.path.join(raw_folder, i_class, "*")
    count = 0
    # time_ = 0
    count_ = 0

for i_class in config.class_names:
    read_path = os.path.join(raw_folder, i_class, "*")
    count = 0
    # time_ = 0
    for file in glob.glob(read_path):
        image_org = cv2.imread(file)
        image = image_org.copy()
        image = cv2.resize(image, dsize=(config.image_size, config.image_size))
        image = np.expand_dims(image, axis=0)
        start = time.time()
        predict = my_model.predict(image, verbose=0)
        end = time.time()
        if(le.inverse_transform([np.argmax(predict)])[0]==i_class):
            count+=1
            count_+=1
            time__ += (end - start)
    print("{}: {:.2f}%".format(i_class,(count/50)*100))
print("Accuracy : {:.2f}%".format((count_/450)*100))
print("Processing time : {:.2f}".format(time__/450))