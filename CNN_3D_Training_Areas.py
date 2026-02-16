import time
import tensorflow
import os
import warnings
import numpy as np
from numpy import newaxis
from keras.models import Sequential
from keras.layers import Flatten
from keras.layers.convolutional import Conv3D
from keras.layers.convolutional import MaxPooling3D
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
warnings.filterwarnings("ignore") #Hid, activation='relu'e messy Numpy warnings
# Open File to write results
# File1 = open('Traj3_CNN3D_Regression_MaxAnt.csv','w')
# File2 = opedf4n('Traj2_BiLSTM_5tCNN_NeighProb_Norm_Filtered_4Decimesteps.csv','w') 

def rmse(y_true, y_pred):
        return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))

# Norm_Factor = 25
# Parameter -------------------------------------------------------
NUM_DT = 3

num_timestep = 3
num_output = 2
epochs_num  = 50
# [samples][pixels][width][height]

ImageWidth = 90
ImageHeight = 90
ImageChannel = 1

# Read data from 3 areas ---------------
print('> Loading data... ')
for ii in xrange(NUM_DT): 
    print("TRAINING DATABASE: ",ii)

    if ii == 0: # Database 1
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

    # Build a model 
    model = Sequential()
    model.add(Conv3D(3, (5, 5, 5), padding='same', input_shape=(num_timestep, ImageHeight, ImageWidth,ImageChannel), activation='relu'))
    model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.2))

    model.add(Conv3D(5, (5, 5, 5), padding='same', activation='relu')) # 5x90x90@3
    # model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.2))

    model.add(MaxPooling3D(pool_size=(1,3,3), strides = (1,3,3), padding='same'))
    model.add(Conv3D(5, (5, 5, 5), padding='same', input_shape=(5, 5,30, 30), activation='relu'))
    # model.add(keras.layers.BatchNormalization())

    model.add(MaxPooling3D(pool_size=(1,3,3), strides = (1,3,3), padding='same'))
    model.add(Conv3D(5, (5, 5, 5), padding='same', input_shape=(5, 5, 10, 10), activation='relu'))
    # model.add(keras.layers.BatchNormalization())

    model.add(TimeDistributed(Flatten()))
    model.add(TimeDistributed(Dense(500, activation='relu')))
    #model.add(keras.layers.BatchNormalization())
    model.add(Dropout(0.2))
    model.add(TimeDistributed(Dense(num_output)))
    # Compile model

    adam_opt = Adam(lr=0.001)
    model.compile(loss="mse", optimizer= adam_opt, metrics=[rmse])
    # plot_model(model, to_file='CNN3D_model.png')

    # model.load_weights("CNN_3D_15Jan_MaxAnt.h5")
    if ii == 0: # Database 1
        model.load_weights("Database1.h5")
    if ii == 1: # Database 2
        model.load_weights("Database2.h5")
    if ii == 2: # Database 3
        model.load_weights("Database3.h5")

    if epochs_num > 0:
        for ep in xrange(epochs_num):
            print("Iteration {} ---".format(ep))
            
            model.fit(input_training,output_training, epochs=1, batch_size=128, verbose=1)
            if ii == 0: # Database 1
                model.save_weights("Database1.h5")
            if ii == 1: # Database 2
                model.save_weights("Database2.h5")
            if ii == 2: # Database 3
                model.save_weights("Database3.h5")

       
