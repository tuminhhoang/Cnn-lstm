import time
import tensorflow
import os
import warnings
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.utils import np_utils
from keras.layers.core import Dense, Activation, Dropout
import matplotlib.pyplot as plt
import pandas as pd
from keras import backend as K
from keras.utils import plot_model
from keras import optimizers
from keras.optimizers import Adam
import keras

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' #Hide messy TensorFlow warnings
warnings.filterwarnings("ignore") #Hide messy Numpy warnings

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Parameter -------------------------------------------------------
NUM_DT = 3 # Number of Database

num_timestep = 3
num_output = 2
epochs_num  = 500

# LSTM -----------------
hidden_layer1 = 250
hidden_layer2 = 200

# [samples][pixels][width][height]
ImageWidth = 90
ImageHeight = 90
ImageChannel = 1
num_features = ImageWidth*ImageHeight*ImageChannel
# Read data --------------------------------------------------
# Read data from 3 areas ---------------
print('> Loading data... ')
for ii in xrange(NUM_DT): 
    print("TRAINING DATABASE: ",ii)

    if ii == 0: # Database 1num_features
        #### Input ##########################
        df_in1=pd.read_csv('Input_CNN3D_Dt1.csv')
        df_in1= np.asarray(df_in1)
        df_combine = df_in1
        input_training = df_combine.reshape(len(df_combine)/ImageHeight,num_timestep, ImageHeight, ImageWidth,ImageChannel)
        print(input_training.shape)

        #### Output ##################################
        df_out1=pd.read_csv('Output_3D_Dt1_NoNorm.csv')
        df_out1= np.asarray(df_out1)
        df_out_combine = df_out1
        output_training = df_out_combine.reshape(len(df_out_combine),num_timestep,num_output)
        print(output_training.shape)
        # print(output_training[0,:,:])

    if ii == 1: # Database 2
        #### Input ##########################
        df_in1=pd.read_csv('Input_CNN3D_Dt2.csv')
        df_in1= np.asarray(df_in1)
        df_combine = df_in1
        input_training = df_combine.reshape(len(df_combine)/ImageHeight,num_timestep, ImageHeight, ImageWidth,ImageChannel)
        print(input_training.shape)

        #### Output ##################################
        df_out1=pd.read_csv('Output_3D_Dt2_NoNorm.csv')
        df_out1= np.asarray(df_out1)
        df_out_combine = df_out1
        output_training = df_out_combine.reshape(len(df_out_combine),num_timestep,num_output)
        print(output_training.shape)

    if ii == 2: # Database 3
        #### Input #########Output_CNN3D_Dt3#################
        df_in1=pd.read_csv('Input_CNN3D_Dt3.csv')
        df_in1= np.asarray(df_in1)
        df_combine = df_in1
        input_training = df_combine.reshape(len(df_combine)/ImageHeight,num_timestep, ImageHeight, ImageWidth,ImageChannel)
        print(input_training.shape)

        #### Output #########################2#########
        df_out1=pd.read_csv('Output_3D_Dt3_NoNorm.csv')
        df_out1= np.asarray(df_out1)
        df_out_combine = df_out1
        output_training = df_out_combine.reshape(len(df_out_combine),num_timestep,num_output)
        print(output_training.shape)

    #######################################################################
    # Build a model 1 ---------------------------------
    # CNN 2D ------------------------------------------
    model1 = Sequential()
    model1.add(TimeDistributed(Conv2D(5, (5, 5), padding='same', input_shape=(ImageHeight, ImageWidth,ImageChannel), activation='relu'), input_shape=(num_timestep, ImageHeight, ImageWidth,ImageChannel)))

    model1.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides = (3,3), padding='same')))
    model1.add(TimeDistributed(Conv2D(5, (5, 5), padding='same', input_shape=(5, 30, 30), activation='relu')))

    model1.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides = 3, padding='same')))
    model1.add(TimeDistributed(Conv2D(5, (5, 5), padding='same', input_shape=(5, 10, 10), activation='relu')))
    
    model1.add(TimeDistributed(MaxPooling2D(pool_size=(2, 2), strides = 2, padding='same')))
    model1.add(TimeDistributed(Conv2D(10, (5, 5), padding='same', input_shape=(10, 5, 5), activation='relu')))

    model1.add(TimeDistributed(keras.layers.BatchNormalization()))
    model1.add(TimeDistributed(Flatten()))

    # LSTM ------------------------------------------
    model1.add(LSTM(hidden_layer1, input_shape=(num_timestep, 250), return_sequences=True))
    model1.add(Dropout(0.2))
  #  model1.add(LSTM(hidden_layer2, return_sequences=True))
  #  model1.add(Dropout(0.2))
    model1.add(TimeDistributed(Dense(num_output)))
    # Compile model
    model1.summary()
    adam_opt = Adam(lr=0.001)
    model1.compile(loss="mse", optimizer= adam_opt, metrics=[rmse])

  #  if ii == 0: # Database 1
  #      if ii == 0: # Database 1
  #          model1.load_weights("CNN_LSTM_Model1_D1.h5")
  #      if ii == 1: # Database 2
  #          model1.load_weights("CNN_LSTM_Model1_D2.h5")
  #      if ii == 2: # Database 3
  #          model1.load_weights("CNN_LSTM_Model1_D3.h5")

    if epochs_num > 0:
        for ep in xrange(epochs_num):
            print("Iteration {} ---".format(ep))
            
            model1.fit(input_training,output_training, epochs=1, batch_size=128, verbose=1)
            if ii == 0: # Database 1
                model1.save_weights("CNN_LSTM_Model1_D1_update.h5")
            if ii == 1: # Database 2
                model1.save_weights("CNN_LSTM_Model1_D2_update.h5")
            if ii == 2: # Database 3
                model1.save_weights("CNN_LSTM_Model1_D3_update.h5")

