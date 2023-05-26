import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from tensorflow.keras.models import load_model
# model = load_model('files/model_adv.h5')

from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
  model2 = load_model("files/model.h5")

import numpy as np
from tensorflow.keras.preprocessing import image
import cv2
from glob import glob

import tensorflow as tf

ip_path = 'final_result/input'
op_path = 'final_result/output'

for i in os.listdir(ip_path):
  img = image.load_img(ip_path+"/"+i, target_size = (256, 256))
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
#   p = model.predict_classes(img)
  # p = (model.predict(img) > 0.5).astype("int32")
  # if(p[0][0] == 1):
  #   print("Patient is Normal. :)")
  # elif(p[0][0] == 0):
  #   print("Patient is affected with covid 19. :(")
  # else:
  #   print("Error occured. Try again!")

images = sorted(glob(os.path.join(ip_path, "*.png")))
# for i in images:
#   print(i)
for i in images:
  ori_x = cv2.imread(i, cv2.IMREAD_COLOR)
  ori_x = cv2.resize(ori_x, (256, 256))
  x = ori_x/255.0
  x = x.astype(np.float32)
  x = np.expand_dims(x, axis=0)
  ori_y = cv2.imread(i, cv2.IMREAD_GRAYSCALE)
  ori_y = cv2.resize(ori_y, (256, 256))
  ori_y = np.expand_dims(ori_y, axis=-1)  ## (512, 512, 1)
  ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)  ## (512, 512, 3)
  y_pred = model2.predict(x)[0] > 0.5
  y_pred = y_pred.astype(np.int32)
  save_image_path = op_path+"/output.png"
  y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
  y_pred = y_pred*255
  color = np.array([255, 0, 0]) 
  result = ori_x.copy()
  # result[y_pred > 0] = y_pred[y_pred > 0]
  mask_indices = np.any(y_pred, axis=2)
  result[mask_indices] = color
  cv2.imwrite(save_image_path, result)