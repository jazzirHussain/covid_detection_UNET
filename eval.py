import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
import numpy as np
import cv2
from glob import glob
from tqdm import tqdm
import tensorflow as tf
from tensorflow.keras.utils import CustomObjectScope
from metrics import dice_loss, dice_coef, iou
from train import load_data, create_dir, tf_dataset

H = 256
W = 256

if __name__ == "__main__":
    """ Seeding """
    np.random.seed(42)
    tf.random.set_seed(42)

    """ Directory for storing files """
    create_dir("results")

    """ Loading model """
    with CustomObjectScope({'iou': iou, 'dice_coef': dice_coef, 'dice_loss': dice_loss}):
        model = tf.keras.models.load_model("files/model.h5")

    """ Dataset """
    dataset_path = "dataset"
    (train_x, train_y), (valid_x, valid_y), (test_x, test_y) = load_data(dataset_path)

    """ Predicting the mask """
    for x, y in tqdm(zip(test_x, test_y), total=2):
        """ Extracing the image name. """
        image_name = x.split("\\")[-1]
        print(image_name)
        """ Reading the image """
        ori_x = cv2.imread(x, cv2.IMREAD_COLOR)
        ori_x = cv2.resize(ori_x, (W, H))
        x = ori_x/255.0
        x = x.astype(np.float32)
        x = np.expand_dims(x, axis=0)

        """ Reading the mask """
        ori_y = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        # ori_y2 = cv2.imread(y2, cv2.IMREAD_GRAYSCALE)
        # ori_y = ori_y1 + ori_y2
        # print(np.shape(ori_y))
        ori_y = cv2.resize(ori_y, (W, H))
        ori_y = np.expand_dims(ori_y, axis=-1)  ## (512, 512, 1)
        ori_y = np.concatenate([ori_y, ori_y, ori_y], axis=-1)  ## (512, 512, 3)

        """ Predicting the mask. """
        y_pred = model.predict(x)[0] > 0.5
        y_pred = y_pred.astype(np.int32)
        """ Saving the predicted mask along with the image and GT """
        save_image_path = f"results/{image_name}"
        y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
        y_pred = y_pred*255

        sep_line = np.ones((H, 10, 3)) * 255

        cat_image = np.concatenate([ori_x, sep_line, ori_y, sep_line, y_pred*255], axis=1)
      
        # image2 = y_pred.astype(float) / 255.0

        # Apply the mask to image1
        color = np.array([255, 0, 0]) 
        result = ori_x.copy()
        # result[y_pred > 0] = y_pred[y_pred > 0]
        mask_indices = np.any(y_pred, axis=2)
        result[mask_indices] = color
        cv2.imwrite(save_image_path, result)
        import matplotlib.pyplot as plt

        plt.subplot(1, 3, 1)
        plt.imshow(ori_x)
        plt.title('Image 1')

        plt.subplot(1, 3, 2)
        plt.imshow(y_pred)
        plt.title('Image 2 (Mask)')

        plt.subplot(1, 3, 3)
        plt.imshow(result)
        plt.title('Result')

        plt.show()
        break







