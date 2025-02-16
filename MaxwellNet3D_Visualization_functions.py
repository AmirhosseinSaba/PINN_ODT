# -*- coding: utf-8 -*-
"""
Created on Mon Mar 27 18:21:15 2022

@author: Amirhossein Saba  -- MaxwellNet Version 3D
"""
#%%
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from matplotlib import cm


#%%
def generates(model, RI_input, MaxwellNet_Loss_class, Physical_attributes):
  prediction    = model(RI_input, training=False)  
  RI_input      = tf.cast(RI_input,tf.complex64)
  
  losss         = MaxwellNet_Loss_class(Physical_attributes).Maxwell_loss(RI_input, prediction)
  prediction    = prediction.numpy()

  plt.figure(figsize=(4,4))
  plt.title("Loss= {}".format(losss))
  plt.imshow(np.real((prediction[0,RI_input.shape[1]//2,:,:,0]+1j*prediction[0,RI_input.shape[1]//2,:,:,1])))
  plt.colorbar()
  plt.axis('off')
  plt.show()
  
#%%
def Compare_COMSOL(model, RI_set, indx, Field_COMSOL, cmin, cmax, mode):
    RII = RI_set[indx:indx+1,:,:,:]
    prediction  = model(RII, training=False).numpy()
    prediction = (prediction[:,:,:,0] +1j*prediction[:,:,:,1]).reshape((256,256))
    RII = RII.reshape((256,256))
    if mode == 'abs':
        Display_Field_COMSOL = np.fliplr(np.rot90(np.abs(Field_COMSOL),3))
        Display_Field_DNN= np.abs(prediction)
    elif mode == 'real':
        Display_Field_COMSOL = np.fliplr(np.rot90(np.real(Field_COMSOL),3))
        Display_Field_DNN= np.real(prediction)        
    elif mode == 'imag':
        Display_Field_COMSOL = np.fliplr(np.rot90(np.imag(Field_COMSOL),3))
        Display_Field_DNN= np.imag(prediction)  
         
    f, axs = plt.subplots(2, 3) 
    title_ = "MaxwellNet"
    ax1 = axs[0, 0]
    ax1.set_title(title_)
    ax1.imshow(RII)
    ax1.axis('off')
  
    title_ = "COMSOL"
    ax1 = axs[0, 1]
    ax1.set_title(title_)
    ax1.imshow(RII)
    ax1.axis('off')
    
    title_ = "Diff"
    ax1 = axs[0, 2]
    ax1.set_title(title_)
    ax1.imshow(RII-RII)
    ax1.axis('off')
    
    ax1 = axs[1, 0]
    ax1.imshow(Display_Field_DNN, vmin = cmin, vmax = cmax)
    ax1.axis('off')
    
    ax1 = axs[1, 1]
    ax1.imshow(Display_Field_COMSOL, vmin = cmin, vmax = cmax)
    ax1.axis('off')
    
    ax1 = axs[1, 2]
    ax1.imshow(Display_Field_COMSOL-Display_Field_DNN, vmin = cmin, vmax = cmax)
    ax1.axis('off')

#%%

def my_plot3D(ax1, Field, Physical_attributes, PML_tickk, vmin, vmax, order='ZXY'):
    if order == 'ZXY':
        Field = Field
    elif order == 'XYZ':
        Field = np.transpose(Field,[1,2,0])
    elif order == 'YXZ':
            Field = np.transpose(Field,[2,1,0]) 
    
    X, Y, Z         = Physical_attributes.Computational_domain()
    
    center_pix_spa  = X.shape[0]//2-PML_tickk  
    center_pix      = X.shape[0]//2
    xxx = X[center_pix,PML_tickk:64-PML_tickk,PML_tickk:64-PML_tickk]
    yyy = Y[PML_tickk:64-PML_tickk,PML_tickk:64-PML_tickk,center_pix]
    zzz = Z[center_pix,PML_tickk:64-PML_tickk,PML_tickk:64-PML_tickk]
    Field11 = Field[center_pix,PML_tickk:64-PML_tickk,PML_tickk:center_pix]
    Field12 = Field[center_pix,PML_tickk:64-PML_tickk,center_pix:-1-PML_tickk]
    Field2 = Field[PML_tickk:64-PML_tickk,PML_tickk:64-PML_tickk,center_pix]
    ax1.contourf(zzz[:,0:center_pix_spa], xxx[:,0:center_pix_spa], Field11, 100, zdir='z', offset=0.0, cmap=cm.coolwarm, vmin = vmin, vmax= vmax)
    ax1.contourf(Field2, np.transpose(yyy), xxx, 100, zdir='x', offset=0.0, cmap=cm.coolwarm, vmin = vmin, vmax= vmax)
    ax1.contourf(zzz[:,center_pix_spa:-1], xxx[:,center_pix_spa:-1], Field12, 100, zdir='z', offset=0.0, cmap=cm.coolwarm, vmin = vmin, vmax= vmax)
#%%        