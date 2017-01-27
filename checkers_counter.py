import cv2
import numpy as np
import csv
import os
import Image
from keras.models import Model
from keras.layers import Input, merge, Convolution2D, MaxPooling2D, UpSampling2D
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint, LearningRateScheduler
from keras import backend as K
import matplotlib.pyplot as plt

K.set_image_dim_ordering('th')

img_rows = 64
img_cols = 80

smooth = 1.

def get_data(path):
    """
    Get only 100 images for local testing
    """
    X=[]
    checkers=[]
    filenames=[]
    y=[]

    with open('train_clean_checkers.csv','r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            checkers.append(row['checkers'])
            filenames.append(row['filename'])
        count_check=0
        for line in checkers:
            if count_check<=99:

                line=line.replace("-","").split(":")
                line.pop(0)
                line.pop(-1)
                line_int=[]
                for i in line:
                    line_int.append(int(i))
                count_check+=1

                y.append(line_int)
        data_path = os.path.join(path,'train_clean_checkers')
        data_folder = os.listdir(data_path)

        count_file=0
        for file in data_folder:
            if count_file <= 99:

                img_path = "%s/%s" %(data_path, file)
                img=cv2.imread(img_path)
                img = cv2.resize(img, (img_cols, img_rows), interpolation=cv2.INTER_CUBIC)
                img = img.reshape( 3, img_rows, img_cols)
                X.append(img)
                count_file+=1



    print "**"*20 
    print "data loaded"
    print "**"*20


    return np.array(X),np.array(y),filenames



def get_unet():
    inputs = Input((3, img_rows, img_cols))
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation='relu', border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation='relu', border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation='relu', border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation='relu', border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation='relu', border_mode='same')(conv9)

    conv10 = Convolution2D(1, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)

    model.compile(optimizer=Adam(lr=1e-5), loss=dice_coef_loss, metrics=[dice_coef])

    return model

def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    return -dice_coef(y_true, y_pred)

def validation(model,X,y, num_epoch):
    np.random.seed(60)
    np.random.shuffle(X)
    np.random.seed(60)
    np.random.shuffle(y)

    num_test = int(X.shape[0]*0.8)

    X_test = X[:num_test]
    y_test = y[:num_test]

    X_valid = X[num_test:]
    y_valid = y[num_test:]

    model.fit(X_test,y_test, nb_epoch=num_epoch, batch_size=64)

    score = model.evaluate(X_valid, y_valid)

    return score


if __name__ == "__main__":
    X, y, filenames = get_data('./')
    model = get_unet()

    validation(model, X, y, 20)
    
