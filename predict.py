#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import warnings
warnings.filterwarnings("ignore")

import numpy as np
from keras.models import load_model
from keras.preprocessing import image

class birds:
    def __init__(self,filename):
        self.filename =filename


    def predictionbirds(self):
        # load model
        model = load_model('model.h5')

        # summarize model
        #model.summary()
        imagename = self.filename
        test_image = image.load_img(imagename, target_size = (150, 150))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        result = model.predict(test_image)

        if result[0][0] == 1:
            prediction = 'Crow'
            return [{"image": prediction}]
            #print(result)
            print(prediction)
        if result[0][1] == 1:
            prediction = 'Parrot'
            return [{"image": prediction}]
            #print(result)
            print(prediction)
        if result[0][2] == 1:
            prediction = 'Peacock'
            return [{"image": prediction}]
            #print(result)
            print(prediction)
        if result[0][3] == 1:
            prediction = 'Pegion'
            return [{"image": prediction}]
            #print(result)
            print(prediction)
        if result[0][4] == 1:
            prediction = 'Sparrow'
            return [{"image": prediction}]
            #print(result)
            print(prediction)


