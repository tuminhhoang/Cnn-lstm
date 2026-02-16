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

#####################################################
# Function Definition
#####################################################
def Get_Locations(OutLoc, First_Loc, PredictedArray, num_timestep,File1):
    # Delta 1 = L2 - L1
    # Delta 2 = L3 - L2
    # Delta 3 = L3 - L1
    OutLoc[0,0] = First_Loc[0]
    OutLoc[0,1] = First_Loc[1]
    for ii in xrange(1, num_timestep):
        OutLoc[ii,0] = PredictedArray[0, ii-1, 0]
        OutLoc[ii,1] = PredictedArray[0, ii-1, 1] 
    # Write to Files
    for ii in xrange(num_timestep):
        File1.write(str(PredictedArray[0, ii, 0]) + ',') 
        File1.write(str(PredictedArray[0, ii, 1]) + ',') 
    File1.write('\n') 

# Parameter -------------------------------------------------------
num_timestep = 3
num_output = 2

# LSTM -----------------
hidden_layer1 = 200
hidden_layer2 = 200

# [samples][pixels][width][height]
ImageWidth = 90
ImageHeight = 90
ImageChannel = 1
num_features = ImageWidth*ImageHeight*ImageChannel
# Number of neighbour Locations:
K_Neigh = 3

# Maximum Accepted Distance
MaxDistance = 4

# RSSI Threshold of each area
Thresh1_High = 0  # Area 1
Thresh1_Low = -59
Thresh2_Low = -100
Thresh2_High = -59 # Area 3, 4

# Center Array 
Center1 = [4.0, 0.5]
Center2 = [6.2, 1.5]
Center3 = [14, 3.7]
Center4 = [5.1, -7.5]

#####################################################
# Test ##############################################
#####################################################
# Open File to write results
File1 = open('CNN_Traj1_Day1_output.csv','w')

print('> Loading CSI Test... ')
df4=pd.read_csv('Norm_Am_CNN_Traj1_Day1.csv')
df4= np.asarray(df4)
input_test = df4
#.reshape(len(df4)/ImageHeight,ImageHeight, ImageWidth,ImageChannel)
print(input_test.shape)

print('> Loading RSSI Test... ')
df5=pd.read_csv('RSSI_Traj1_Day1.csv')
df5= np.asarray(df5)
rssi_test = df5.reshape(len(df5), 3)
print(rssi_test.shape)

print('> Loading Test Locations... ')
df6=pd.read_csv('Locations_Traj1_Day1.csv')
df6= np.asarray(df6)
true_test = df6.reshape(len(df6),2)
print(true_test.shape)

NumTestPoints = len(df6)

#######################################################################
#           BUILD MODEL 
#######################################################################
# Build a model 1 ---------------------------------
model1 = Sequential()
model1.add(TimeDistributed(Conv2D(10, (5, 5), padding='same', input_shape=(ImageHeight, ImageWidth,ImageChannel), activation='relu'), input_shape=(num_timestep, ImageHeight, ImageWidth,ImageChannel)))

model1.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides = (3,3), padding='same')))
model1.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', input_shape=(16, 30, 30), activation='relu')))

model1.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides = 3, padding='same')))
model1.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', input_shape=(16, 10, 10), activation='relu')))
model1.add(TimeDistributed(keras.layers.BatchNormalization()))
model1.add(TimeDistributed(Flatten()))

# LSTM ------------------------------------------
model1.add(LSTM(hidden_layer1, input_shape=(num_timestep, 1600), return_sequences=True))
model1.add(Dropout(0.2))
#  model1.add(LSTM(hidden_layer2, return_sequences=True))
#  model1.add(Dropout(0.2))
model1.add(TimeDistributed(Dense(num_output)))
# Compile model
model1.summary()
adam_opt = Adam(lr=0.001)
model1.compile(loss="mse", optimizer= adam_opt, metrics=[rmse])

######################################################################
# Build a model 2 
model2 = Sequential()
model2.add(TimeDistributed(Conv2D(10, (5, 5), padding='same', input_shape=(ImageHeight, ImageWidth,ImageChannel), activation='relu'), input_shape=(num_timestep, ImageHeight, ImageWidth,ImageChannel)))

model2.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides = (3,3), padding='same')))
model2.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', input_shape=(16, 30, 30), activation='relu')))

model2.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides = 3, padding='same')))
model2.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', input_shape=(16, 10, 10), activation='relu')))
model2.add(TimeDistributed(keras.layers.BatchNormalization()))
model2.add(TimeDistributed(Flatten()))

# LSTM ------------------------------------------
model2.add(LSTM(hidden_layer1, input_shape=(num_timestep, 1600), return_sequences=True))
model2.add(Dropout(0.2))
#  model1.add(LSTM(hidden_layer2, return_sequences=True))
#  model1.add(Dropout(0.2))
model2.add(TimeDistributed(Dense(num_output)))
# Compile model
model2.summary()
model2.compile(loss="mse", optimizer= adam_opt, metrics=[rmse])

######################################################################
# Build a model 2 
model3 = Sequential()
model3.add(TimeDistributed(Conv2D(10, (5, 5), padding='same', input_shape=(ImageHeight, ImageWidth,ImageChannel), activation='relu'), input_shape=(num_timestep, ImageHeight, ImageWidth,ImageChannel)))

model3.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides = (3,3), padding='same')))
model3.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', input_shape=(16, 30, 30), activation='relu')))

model3.add(TimeDistributed(MaxPooling2D(pool_size=(3, 3), strides = 3, padding='same')))
model3.add(TimeDistributed(Conv2D(16, (5, 5), padding='same', input_shape=(16, 10, 10), activation='relu')))
model3.add(TimeDistributed(keras.layers.BatchNormalization()))
model3.add(TimeDistributed(Flatten()))

# LSTM ------------------------------------------
model3.add(LSTM(hidden_layer1, input_shape=(num_timestep, 1600), return_sequences=True))
model3.add(Dropout(0.2))
#  model1.add(LSTM(hidden_layer2, return_sequences=True))
#  model1.add(Dropout(0.2))
model3.add(TimeDistributed(Dense(num_output)))
# Compile model
model3.summary()
model3.compile(loss="mse", optimizer= adam_opt, metrics=[rmse])

######################################################################
# Load Model
model1.load_weights("CNN_LSTM_Model1_D1.h5")
model2.load_weights("CNN_LSTM_Model1_D2.h5")
model3.load_weights("CNN_LSTM_Model1_D3.h5")

#######################################################################
#           TEST 
#######################################################################
print('> Testing ... ')
TestingTraj = np.zeros((ImageHeight, ImageWidth*ImageChannel*num_timestep))
First_Loc = true_test[0,:] 

##########################################################
##### Init the first 3 locations:
#########################################################
#CntRow = 0
for ii in xrange(num_timestep):
    for jj in xrange(ImageWidth*ImageChannel):
        TestingTraj[:,ii*ImageWidth*ImageChannel+jj] = input_test[ii*ImageHeight:(ii+1)*ImageHeight,jj] # Testing
        # CntRow = CntRow + 1

print(TestingTraj.shape)
TestingInput = TestingTraj.reshape(1,num_timestep, ImageHeight, ImageWidth,ImageChannel)

##########################################################
##### Test for the rest locations
#########################################################
error = np.zeros(NumTestPoints)
Average_Err = 0 
for CntTest in xrange(num_timestep,NumTestPoints):
    print("Test {} ---------- ".format(CntTest+1))

    # ------------------------------------------
    ##### RSSI to divide area #####################
    # ------------------------------------------
    RSSI_Average = 0
    for ii in xrange(num_timestep):
        RSSI_Average = RSSI_Average + rssi_test[CntTest - ii -1,2]
    RSSI_Average = RSSI_Average/num_timestep
    print(RSSI_Average)

    Area_Choice = np.zeros(3) # 3 Areas
    # Find the List of each area 
    if (RSSI_Average < Thresh1_High) and (RSSI_Average > Thresh1_Low):
        Area_Choice[0] = 1
    if (RSSI_Average <= Thresh2_High) and (RSSI_Average > Thresh2_Low):
        # Compare Previous Location with the Center of Area 3
        Dis3 = np.power((First_Loc[0] - Center3[0]),2) + np.power((First_Loc[1] - Center3[1]),2)
        # Compare Previous Location with the Center of Area 4
        Dis4 = np.power((First_Loc[0] - Center4[0]),2) + np.power((First_Loc[1] - Center4[1]),2)
        if Dis3 < Dis4:
            Area_Choice[1] = 1
        else:
            Area_Choice[2] = 1

    # --------------------------num_timestep----------------
    ##### Prediction #####################
    # ------------------------------------------
    OutLoc = np.zeros((num_timestep,2))
    if Area_Choice[0] == 1: # Area 1
        PredictedArray = model1.predict(TestingInput)    # Prediction 
        print "Database 1 ... "
        print(PredictedArray)
        
    if Area_Choice[1] == 1: # Area 2
        print "Database 2 ... "
        PredictedArray = model2.predict(TestingInput)    # Prediction  
        print(PredictedArray)

    if Area_Choice[2] == 1: # Area 3
        print "Database 3 ... "
        PredictedArray = model3.predict(TestingInput)    # Prediction 
        print(PredictedArray)
    
    Get_Locations(OutLoc, First_Loc, PredictedArray, num_timestep, File1) 
    # print(OutLoc)
    # Update the First_Loc
    # First_Loc = OutLoc[1,:] # Update the First_Loc as the 2nd output location 
    First_Loc =  true_test[CntTest-2, :]
    # ------------------------------------------
    ##### Calculate Error #####################
    # ------------------------------------------
    Correct_L = true_test[CntTest-1, :]
    error[CntTest-1] = np.sqrt(np.power((OutLoc[2,0] - Correct_L[0]),2)+np.power((OutLoc[2,1] - Correct_L[1]),2))
    print "Predict: {}--- Exact: {} , Error: {}" .format((OutLoc[2,0], OutLoc[2,1]), (Correct_L[0], Correct_L[1]),  error[CntTest-1])
    Average_Err = Average_Err + error[CntTest-1]

    # ------------------------------------------
    ##### Update Testing Trajectory ############
    # ------------------------------------------
    for ii in xrange(num_timestep-1): # Arrange the Buffer
        for jj in xrange(ImageWidth*ImageChannel):
            TestingTraj[:,ii*ImageWidth*ImageChannel+jj] = TestingTraj[:,(ii+1)*ImageWidth*ImageChannel+jj] # Testing

    # Update the final slot 
    for jj in xrange(ImageWidth*ImageChannel):
        TestingTraj[:, (num_timestep-1)*ImageWidth+jj] = input_test[CntTest*ImageHeight:(CntTest+1)*ImageHeight,jj]  # Testing
    TestingInput = TestingTraj.reshape(1,num_timestep, ImageHeight, ImageWidth,ImageChannel)

Average_Err = Average_Err/NumTestPoints
print " ** Average Error: ", Average_Err

Std_Err = 0
for k in xrange(NumTestPoints):
    Std_Err = Std_Err +  np.power((error[k] - Average_Err),2)
Std_Err = Std_Err/NumTestPoints
Std_Err = np.sqrt(Std_Err)
print "** Std: ", Std_Err 

File1.close()
