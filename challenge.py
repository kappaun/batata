# -*- coding: utf-8 -*-
'''
Interview Task Code
Andressa Kappaun
'''
import os
import numpy as np
import zipfile
import csv

from keras.models import Sequential
from keras.layers import Dense, Activation

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from keras.optimizers import SGD
from keras.regularizers import l2
from keras.models import load_model


def get_data(path):
    '''
    Function recives the path of = files
    And returns data as np arrays

    Args:
      path (str): Same folder of this code

    Returns:
      Np.Array of X Train data, 
      Np.Array of y Train data,
      Np.Array of Test data
    
    '''
    X = []
    y = []
    X_test = []

    train_path = os.path.join(path, 'train_100k.zip')
    train_folder = zipfile.ZipFile(train_path, 'r')

    train_100k = train_folder.open('train_100k.csv')
    train_100k_gt = train_folder.open('train_100k.truth.csv')

    test_path = os.path.join(path, 'test_100k.zip')
    test_folder = zipfile.ZipFile(test_path, 'r')
    test_100k = test_folder.open('test_100k.csv')

    train_100k.readline()
    for line in train_100k.readlines():
        data = [float(d) for d in line.split(",")[1:]]
        X.append(data)

    train_100k_gt.readline()
    for line in train_100k_gt.readlines():
        data_gt = [float(d) for d in line.split(",")[1:]]
        y.append(data_gt)

    test_100k.readline()
    for line in test_100k.readlines():
        data_test = [float(d) for d in line.split(",")[1:]]
        X_test.append(data_test)

    return np.array(X), np.array(y), np.array(X_test)


def pre_proc(X):
  ''' 
  Compute separatly the arithmetic mean along x and y.
  Scales each feature individually based on the mean range,
  translates in between 0 and 1 then fit to data
  Returns a transformed version of X
  '''
  X[:, 0::2] = X[:, 0::2] - np.mean(X[:, 0::2])
  X[:, 1::2] = X[:, 1::2] - np.mean(X[:, 1::2])
  return MinMaxScaler().fit_transform(X)


def create_model(X, y,
                 hidden_layer_size,
                 num_layers,
                 metric,
                 learning_rate=0.01,
                 activation_func="sigmoid",
                 num_epoch=20,
                 loss="mean_squared_error",
                 input_dim=20):
    '''
    Creates keras sequential model,
    learns with 80% of data
    and validates it with 20% remaining


    Args:
    X (np.array) = Data
    y (np.array) = Label data
    hidden_layer_size (int) = Size of the hidden layer of the network
    metric (function) = function to judge the performance of the model
    learning_rate (float) = learning rate >=0.0, 0.01 on this case
    activation_func (function) = activation function to be applied to an output
    num_epoch (int) = number of epochs to train the model
    loss (function) = objective function to compile the model
    input_dim (int) = Dimentionality of the input

    Returns:
    constructed model and model evaluation as score
    '''

    model = Sequential()
    layers_size = [2**i for i in xrange(num_layers, -1, -1)]

    model.add(Dense(output_dim=layers_size[0]*hidden_layer_size, input_dim=input_dim))
    model.add(Activation(activation_func))

    for layer_size in layers_size[1:-1]:
        model.add(Dense(output_dim=layer_size*hidden_layer_size))
        model.add(Activation(activation_func))

    model.add(Dense(output_dim=1))

    sgd = SGD(lr=learning_rate,
              decay=1e-8,
              momentum=0.9)

    model.compile(loss=loss,
                  optimizer=sgd,
                  metrics=metric)

    np.random.seed(666)
    np.random.shuffle(X)
    np.random.seed(666)
    np.random.shuffle(y)

    num_test = int(X.shape[0]*0.8)

    X_test = X[:num_test]
    y_test = y[:num_test]

    X_valid = X[num_test:]
    y_valid = y[num_test:]

    model.fit(X_test, y_test, nb_epoch=num_epoch, batch_size=64)
    score = model.evaluate(X_valid, y_valid)

    return model, score


def apply_model(model_slope, model_interceptor, X_test):
    '''
    Applies created models to predict the test data


    Args:
    model_slope (sequential model)
    model_interceptor,
    X_test = np.array of the test data
    Returns:
    Np array of predictions

    '''

    pred_slope = model_slope.predict(X_test, verbose=0)
    pred_interceptor = model_interceptor.predict(X_test, verbose=0)

    pred_submission = [["id", "slope", "intercept"]]

    for i in range(len(pred_slope)):
        line = [i]
        line.append(float(pred_slope[i]))
        #line.append(1.0)
        line.append(float(pred_interceptor[i]))
        pred_submission.append(line)

    pred_file = open("submission.csv", 'w')
    wr = csv.writer(pred_file)
    wr.writerows(pred_submission)

    return pred_submission

if __name__ == "__main__":

    X_train, y_train, X_test = get_data("./")
    X_train_slope = X_train/100.0
    X_test_slope = X_test/100.0

    X_train_interceptor = pre_proc(X_train)
    X_test_interceptor = X_test

    '''
    model_slope, score_slope = create_model(X_train_slope,
                                            y_train[:, 0],
                                           num_layers=4,
                                           hidden_layer_size=64,
                                           metric=["mean_squared_error"],
                                           loss="mean_squared_error")
    '''
    #model_slope.save("./model_slope_"+str(score_slope[0])+".csv")
    model_slope = load_model("./model_slope_0.00827933367845.csv")
    #pred = model_slope.predict(X_train)

    #X_train_interceptor = np.append(X_train_interceptor, pred, axis=1)
   # X_train_interceptor = pre_proc(X_train_interceptor)
    '''
    model_interceptor, score_interceptor = create_model(X_train_interceptor,
                                                      y_train[:, 1],
                                                      hidden_layer_size=256,
                                                      num_layers=2,
                                                      loss="mean_absolute_error",
                                                      metric=["mean_absolute_error"],
                                                      num_epoch=150,
                                                      input_dim=20)

    model_interceptor.save("./model_interceptor_"+str(score_interceptor[0])+".csv")
    '''
    model_interceptor = load_model("./model_interceptor_7.52288659897.csv")

    apply_model(model_slope,model_interceptor,X_train)
    
    print "\n"
    #print "SCORE SLOPE: ", score_slope
    #print "SCORE INTERCEPTOR: ", score_interceptor
