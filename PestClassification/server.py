from flask import Flask, render_template, request
from keras.models import load_model
import os
import cv2
import config

import numpy as np

# Khởi tạo Flask
app = Flask(__name__)

model_vgg16 = load_model(os.path.join("models","VGG16-f-w-max.keras"))

import pickle
file = open('le.pkl','rb')
le = pickle.load(file)
file.close()

@app.route("/", methods=['GET', 'POST'])
def home_page():
    if request.method == "POST":
        try:
            # get file gui len
            image = request.files['file']
            if image:
                #luu file
                path_to_save = os.path.join("test/files",image.filename)
                print("Save = ", path_to_save)
                image.save(path_to_save)

                frame = cv2.imread(path_to_save)
                frame = cv2.resize(frame, dsize = (config.image_size, config.image_size))
                #convert thanh dang tensor
                frame = np.expand_dims(frame, axis=0)

                #run model 
                predict = model_vgg16.predict(frame)
                print("VGG16")

                list_pred = np.argsort(predict)
                more = "{} : {:.2f}%; {} : {:.2f}%; {} : {:.2f}%".format(le.inverse_transform([np.argmax(predict)])[0],predict[0][list_pred[0][-1]]*100,le.inverse_transform([list_pred[0][-2]])[0],predict[0][list_pred[0][-2]]*100,le.inverse_transform([list_pred[0][-3]])[0],predict[0][list_pred[0][-3]]*100)
                predict_name = le.inverse_transform([np.argmax(predict)])[0]

                return render_template("index.html", 
                        image = image.filename, 
                        msg="Upload successful!", 
                        predict_name=predict_name,
                        more=more)
            else:
                return render_template('index.html', msg='Choose file to upload!')
        except Exception as ex:
            print(ex)
            return render_template('index.html', msg="Can not analyze the image!")
    else:
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug = False)