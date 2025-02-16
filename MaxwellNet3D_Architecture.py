# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:14:43 2022

@author: A. Saba
"""

#%%
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Activation, Lambda
from tensorflow.keras.layers import BatchNormalization, Conv3D, AveragePooling3D
from tensorflow.keras.layers import concatenate, Conv3DTranspose

#%%
class MaxwellNet_Architecture:
    def __init__(self, input_size, kernel_size = (3,3,3), Encoder_layers = 5, \
                 activation = 'elu', Pooling = 2, input_nomalization = 0.20):
        self.kernel_size    = kernel_size
        self.Encoder_layers = Encoder_layers        
        self.input_size     = input_size            # [Ny, Nx, Nz, 1]
        self.activation     = activation
        self.Pooling        = Pooling
        self.Normalization  = input_nomalization
        
#% Define Encoder

    def Encoding(self, input_layer, depth):
        down = Conv3D(depth, self.kernel_size, padding='same', kernel_initializer='random_uniform',\
                      bias_initializer='zeros')(input_layer)
        # down = BatchNormalization()(down)
        down = Activation(self.activation)(down)
        down = Conv3D(depth, self.kernel_size, padding='same', kernel_initializer='random_uniform',\
                      bias_initializer='zeros')(down)
        # down = BatchNormalization()(down)
        down_pool = AveragePooling3D((self.Pooling, self.Pooling, self.Pooling), strides=(2, 2, 2))(down)
        return down_pool, down
    
#% Define Decoder
    def Decoding(self, input_layer, skip_con, depth):
        up = Conv3DTranspose(depth, kernel_size = (2, 2, 2), strides=(2, 2, 2), use_bias=False, \
                             padding='same')(input_layer)
        up = concatenate([up, skip_con], axis=4)
        up = Conv3D(depth, self.kernel_size, padding='same',kernel_initializer='random_uniform',bias_initializer='zeros',activation=None)(up)
        up = Activation(self.activation)(up)
        up = Conv3D(depth//2, self.kernel_size, padding='same',kernel_initializer='random_uniform',bias_initializer='zeros')(up)
        return up

#% Define Network

    def UNet(self, depth_start = 16):
        # Input Layer
        inputs  = Input(shape=self.input_size)
        lay1    = Lambda(lambda x: (x)/self.Normalization)(inputs)   
        
        # Encoder section
        down    = {}
        down_pool, down[0] = self.Encoding(lay1, depth_start)
        for l in range(1, self.Encoder_layers+1):
            down_pool, down[l] = self.Encoding(down_pool, depth_start*(2**l))
        
        # Latent section
        center = Conv3D(depth_start*(2**( self.Encoder_layers+1)), self.kernel_size, \
                        padding='same',kernel_initializer='random_uniform')(down_pool)
        center = Activation(self.activation)(center)
        center = Conv3D(depth_start*(2**( self.Encoder_layers)), self.kernel_size, \
                        padding='same',kernel_initializer='random_uniform')(center)
        
        # Decoder section
        up     = self.Decoding(center, down[self.Encoder_layers], depth_start*(2**( self.Encoder_layers)))
        for l in range(1, self.Encoder_layers+1):
            up = self.Decoding(up, down[self.Encoder_layers-l], depth_start*(2**(self.Encoder_layers-l)))
        
        # Output layer
        outputs = Conv3D(2, (1, 1, 1), use_bias=False, activation=None)(up)
        
        UNET    = Model(inputs=inputs, outputs=outputs)
        
        return UNET

#%    
    
