import warnings
warnings.filterwarnings("ignore")
import numpy as np
from keras.models import load_model
from keras.preprocessing import image
model = load_model('model.h5')
test_image = image.load_img('C:\\Users\\Manas\\OneDrive\\Pictures\\Bird Photo\\Train\\Crow\\images.jpg', target_size = (150, 150))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = model.predict(test_image)
print(result[0])
#training_set.class_indices

if result[0][0] == 1:
    prediction = 'Crow'
    #print(result)
    print(prediction)
if result[0][1] == 1:
    prediction = 'Parrot'
    #print(result)
    print(prediction)
if result[0][2] == 1:
    prediction = 'Peacock'
    #print(result)
    print(prediction)
if result[0][3] == 1:
    prediction = 'Pegion'
    #print(result)
    print(prediction)
if result[0][4] == 1:
    prediction = 'Sparrow'
    #print(result)
    print(prediction)

