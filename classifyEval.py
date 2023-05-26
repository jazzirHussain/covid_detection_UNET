from tensorflow.keras.models import load_model

model = load_model('files/model_adv.h5')
import os
TRAIN_PATH = "classification2/train"
VAL_PATH = "classification2/test"

import numpy as np
# from PIL import Image
from tensorflow.keras.preprocessing import image

y_actual = []
y_test = []


for i in os.listdir("classification2/test/Normal"):
  img = image.load_img("classification2/test/Normal/"+i, target_size = (224, 224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
#   p = model.predict_classes(img)
  p = (model.predict(img) > 0.5).astype("int32")
  y_test.append(p[0][0])
  y_actual.append(1)

print(y_actual,y_test)

for i in os.listdir("classification2/test/COVID"):
  img = image.load_img("classification2/test/COVID/"+i, target_size = (224, 224))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
#   p = model.predict_classes(img)
  p = (model.predict(img) > 0.5).astype("int32")
  y_test.append(p[0][0])
  y_actual.append(0)

# print(y_actual,y_test)