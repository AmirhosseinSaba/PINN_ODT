# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:06:46 2022

@author: Amirhossein Saba  -- MaxwellNet Version 3D
"""

#%% import libraries
import numpy as np
import tensorflow as tf
import os
import sklearn.model_selection
import matplotlib.pyplot as plt

from tensorflow.keras.optimizers import Adam

from MaxwellNet3D_PhysicalModel import Physical_properties, MaxwellNet_Loss
from MaxwellNet3D_Architecture import MaxwellNet_Architecture
from MaxwellNet_trainX import train_unet
from MaxwellNet3D_Visualization_functions import generates, my_plot3D

#%%
tf.config.run_functions_eagerly(True)

#%% Load data
# Load dataset of refrcative index distributions

data_path_load      = os.getcwd()

input_data_filename = data_path_load + "/" + "Example1_Refractive_Index_dataset.npy"
RI                  = np.load(input_data_filename)

# Select one of the exmaples which will be plotted during training
RI_test             = RI[-1,:,:,:].reshape((1,RI.shape[1],RI.shape[2],RI.shape[3],1))


# load or assign physical parameters
NN                  = RI.shape[1]
wl                  = 1.030
dx                  = 0.1
n0                  = 1.333

#%% Divide available data to training and validation sets. The split ratio can be adjusted in the following command.

RI                      = np.expand_dims(RI, 4)     # Expand dimension to make the inputs as batch*Ny*Nx*Nz*1
RI_trainset, RI_testset = sklearn.model_selection.train_test_split(RI, test_size = 0.15, shuffle = False)
del RI

batch_size      = 5
with tf.device('/CPU:0'):
    RI_trainset_t   = (tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(RI_trainset))).batch(batch_size)
    RI_testset_t    = (tf.data.Dataset.from_tensor_slices(tf.convert_to_tensor(RI_testset))).batch(batch_size)

#%% Loading the Physical properties class and network architecture class

Physical_prop   = Physical_properties(NN, NN, NN, wl, dx, dx, dx, n0)

MaxwellNet      = MaxwellNet_Architecture([NN,NN,NN,1], (3,3,3), 4, 'elu', 2, 0.20).UNet(16)
# MaxwellNet.summary()

#%%
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate = 0.5e-4,
    decay_steps = 100,
    decay_rate = 0.5,
    staircase = True)
unet_optimizer  = Adam(lr_schedule)
EPOCHS          = 5
batch_size      = 5

#%%
ac_train, ac_test = train_unet(EPOCHS, batch_size, unet_optimizer, MaxwellNet, MaxwellNet_Loss, \
                               Physical_prop, RI_trainset_t, RI_testset_t, RI_test)

#%% Results evaluation
print('Final Training accuracy = {:0.3f}'.format(ac_train[-1]))
print('Final Validation accuracy = {:0.3f}'.format(ac_test[-1]))

plt.semilogy(np.linspace(1, len(ac_train), len(ac_train)), ac_train, np.linspace(1, len(ac_test), len(ac_test)), ac_test, linewidth=2)
plt.legend(['Training', 'Validation'])
plt.ylabel('Loss')
plt.xlabel('Epochs')

generates(MaxwellNet, RI_test, MaxwellNet_Loss, Physical_prop)

Prediction3D_real = MaxwellNet(RI_test).numpy()[0,:,:,:,0]
Prediction3D_imag = MaxwellNet(RI_test).numpy()[0,:,:,:,1]

fig = plt.figure()

ax  = fig.add_subplot(1, 2, 1, projection='3d')
plt.title('Real{Us}')
my_plot3D(ax , Prediction3D_real, Physical_prop, 16, -0.7, 0.2)
ax  = fig.add_subplot(1, 2, 2, projection='3d')
plt.title('Imag{Us}')
my_plot3D(ax , Prediction3D_imag, Physical_prop, 16, -0.5, 1.6)

