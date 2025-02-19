# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:21:15 2022

@author: Amirhossein Saba  -- MaxwellNet Version 3D
"""
#%%
import numpy as np
import tensorflow as tf
import time
import matplotlib as plt

from MaxwellNet3D_Visualization_functions import generates

#%%
# training function using gradient.tape
@tf.function
def train_step_unet(inp, Network, MaxwellNet_Loss_class, Physical_attributes, optimizer):  
    with tf.GradientTape() as tape:
        U_out   = Network(inp, training=True)
        loss    = MaxwellNet_Loss_class(Physical_attributes).Maxwell_loss(inp, U_out)
        
    gradients_of_unet = tape.gradient(loss, Network.trainable_variables)
    optimizer.apply_gradients(zip(gradients_of_unet, Network.trainable_variables))
    return loss

#%% Validation function 
@tf.function
def val_step_unet(inp, Network, MaxwellNet_Loss_class, Physical_attributes):  
    U_out   = Network(inp, training=True)
    loss    = MaxwellNet_Loss_class(Physical_attributes).Maxwell_loss(inp, U_out)
    return loss

#%% Train function for several epochs
def train_unet(epochs, bchsz, optimizer, Network, MaxwellNet_Loss_class,\
               Physical_attributes, training_set, test_set, RI_test):
  
    start=time.time()
    ac=[]
    ac_test=[]
    
    for epoch in range(epochs+1):    
        #Train
        temp_loss = 0
        temp_loss_test = 0
        for step, (x_batch_train) in enumerate(training_set):
             x_batch_train  = tf.cast(x_batch_train, dtype=tf.complex64)
             temp_loss      = temp_loss + train_step_unet(x_batch_train, Network, MaxwellNet_Loss_class,\
                                                          Physical_attributes, optimizer)
        temp_loss = temp_loss/(step+1)
        
        #print()
        #if(epoch%1000==0):
        #  unet_checkpoint.save(file_prefix = unet_prefix)    
    
        for step, (x_batch_test) in enumerate(test_set):
            x_batch_test    = tf.cast(x_batch_test, dtype=tf.complex64)
            temp_loss_test  = temp_loss_test + val_step_unet(x_batch_test, Network, MaxwellNet_Loss_class,\
                                                             Physical_attributes)
        temp_loss_test      = temp_loss_test/(step+1)   
        
        ac.append(temp_loss)
        ac_test.append(temp_loss_test)
        
        if((epoch+1)%20==0):
            #display.clear_output(wait=True)
            print("Epoch: {},\tLoss: {:0.3f}, \tVal_Loss: {:0.3f} ".format(epoch,(temp_loss),temp_loss_test),'Training time = {} (s)'.format(time.time()-start))
            generates(Network, RI_test, MaxwellNet_Loss_class, Physical_attributes)
            
            plt.semilogy(range(0,epoch+1,1),np.array(ac),range(0,epoch+1,1),np.array(ac_test))
            plt.show()
            plt.pause(0.0001)
    print ('Time taken for the training is {:0.2f} sec\n'.format(time.time()-start))
    #unet_checkpoint.save(file_prefix = unet_prefix)
    return([np.array(ac),np.array(ac_test)])

#%%