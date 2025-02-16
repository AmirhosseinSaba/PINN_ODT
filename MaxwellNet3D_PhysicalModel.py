# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 15:14:43 2022

@author: A. Saba
"""

#%%
import math
import numpy as np
import tensorflow as tf
#%% This class defines the physical parameters and methods that are related to
# the physical parameters such as phase modulation, PML definition

class Physical_properties:
    def __init__(self, Ny, Nx, Nz, Wavelength, dy, dx, dz, n0):
        self.Ny     = Ny
        self.Nx     = Nx
        self.Nz     = Nz
        self.wl     = Wavelength
        self.dy     = dy
        self.dx     = dx
        self.dz     = dz
        self.n0     = n0
        self.k0     = 2*math.pi/Wavelength
        self.k      = (self.k0)*n0
    
#%  This method returns numpy arrays X, Y, Z defining the computational domain.
    def Computational_domain(self):
        [XX, YY, ZZ] = np.meshgrid(np.linspace(-self.Nx//2,self.Nx//2-1,self.Nx),\
                                   np.linspace(-self.Ny//2,self.Ny//2-1,self.Ny),\
                                       np.linspace(-self.Nz//2,self.Nz//2-1,self.Nz))
        X = self.dx*XX
        Y = self.dy*YY
        Z = self.dz*ZZ
        
        return X, Y, Z
 
    
#% This method returns coefficinets for the Perfectly Matched Layer (PML).
# It gets the inputs of sigma (PML coefficient), tickness, and the power factor, p.
    def PML(self, sigma = 0.4, PML_tickness_pixels = 16, p = 4):
        
        X, Y, Z     = self.Computational_domain()

        d           = PML_tickness_pixels*self.dx           # PML thickness in (um)
        CDx         = (self.Nx)*(self.dx)                   # Size of Computational_domain
        CDy         = (self.Ny)*(self.dy)                   # Size of Computational_domain
        CDz         = (self.Nz)*(self.dz)                   # Size of Computational_domain
        
        
        sigma_x     = 1+1j*sigma*((np.abs(X)-(CDx/2-d))**p)
        sigma_y     = 1+1j*sigma*((np.abs(Y)-(CDy/2-d))**p)   
        sigma_z     = 1+1j*sigma*((np.abs(Z)-(CDz/2-d))**p)
        
        s_x         = 1*np.double(np.abs(X)<(CDx/2-d))+sigma_x*np.double(np.abs(X)>=(CDx/2-d))
        s_y         = 1*np.double(np.abs(Y)<(CDy/2-d))+sigma_y*np.double(np.abs(Y)>=(CDy/2-d))
        s_z         = 1*np.double(np.abs(Z)<(CDz/2-d))+sigma_z*np.double(np.abs(Z)>=(CDz/2-d))
        s_x         = s_x.reshape((1,self.Ny,self.Nx,self.Nz,1))
        s_y         = s_y.reshape((1,self.Ny,self.Nx,self.Nz,1))
        s_z         = s_z.reshape((1,self.Ny,self.Nx,self.Nz,1))

        coeffx      = 1/s_x
        coeffy      = 1/s_y
        coeffz      = 1/s_z

        coeffx      = tf.cast(tf.convert_to_tensor(coeffx), dtype=tf.complex64)
        coeffy      = tf.cast(tf.convert_to_tensor(coeffy), dtype=tf.complex64)
        coeffz      = tf.cast(tf.convert_to_tensor(coeffz), dtype=tf.complex64)

        return coeffx, coeffy, coeffz

#% This method defines the phase modulaition based the wavelength exp(2*pi/lambda).
    def Phase_modulation(self):
        X, Y, Z          = self.Computational_domain() 
        Phase_modulation = tf.cast(tf.math.exp(+1j*self.k*Z), dtype=tf.complex64)
        
        return tf.reshape(Phase_modulation, ((1, self.Ny, self.Nx, self.Nz, 1)))
    
#%% This class defines the methods requried for training loss of the MaxwellNet

class MaxwellNet_Loss:
    def __init__(self, Physical_attributes):
        self.Physical_attributes = Physical_attributes        

#% define the function for complex multiplication
    def complex_multiplication(self, a, b, return_parts=True):
      rr  = tf.multiply(tf.math.real(a),tf.math.real(b))-tf.multiply(tf.math.imag(a),tf.math.imag(b))
      ii  = tf.multiply(tf.math.real(a),tf.math.imag(b))+tf.multiply(tf.math.imag(a),tf.math.real(b))
      c   = tf.complex(rr,ii)
      if return_parts:
          return c, rr, ii
      else:
          return c

#% The method which returns kernels for derivative calculation. 
# This works only for 5*5 kernels using Yee Grid
    def Kernels(self, kernel_shape):
        # kernel_shape = [5,5,5,1,1]
        # center_element  = 2
        
        d_e             = np.array([0, 1/24, -9/8, +9/8, -1/24])
        d_h             = np.array([1/24, -9/8, +9/8, -1/24, 0])
        ker_dxe         = np.zeros((kernel_shape[0],kernel_shape[1],kernel_shape[2]), dtype = np.float32)
        ker_dye         = np.zeros((kernel_shape[0],kernel_shape[1],kernel_shape[2]), dtype = np.float32)
        ker_dze         = np.zeros((kernel_shape[0],kernel_shape[1],kernel_shape[2]), dtype = np.float32)
        
        center_element  = kernel_shape[0]//2
        ker_dxe[center_element,:,center_element] = d_e
        ker_dye[:,center_element,center_element] = d_e
        ker_dze[center_element,center_element,:] = d_e
        ker_dxh         = np.zeros((kernel_shape[0],kernel_shape[1],kernel_shape[2]), dtype = np.float32)
        ker_dyh         = np.zeros((kernel_shape[0],kernel_shape[1],kernel_shape[2]), dtype = np.float32)
        ker_dzh         = np.zeros((kernel_shape[0],kernel_shape[1],kernel_shape[2]), dtype = np.float32)
        ker_dxh[center_element,:,center_element] = d_h
        ker_dyh[:,center_element,center_element] = d_h
        ker_dzh[center_element,center_element,:] = d_h

        ker_dxe_t       = tf.reshape(tf.convert_to_tensor(ker_dxe),kernel_shape)   #e
        ker_dye_t       = tf.reshape(tf.convert_to_tensor(ker_dye),kernel_shape)   #e
        ker_dze_t       = tf.reshape(tf.convert_to_tensor(ker_dze),kernel_shape)   #e
        ker_dxh_t       = tf.reshape(tf.convert_to_tensor(ker_dxh),kernel_shape)   #h
        ker_dyh_t       = tf.reshape(tf.convert_to_tensor(ker_dyh),kernel_shape)   #h 
        ker_dzh_t       = tf.reshape(tf.convert_to_tensor(ker_dzh),kernel_shape)   #h
        return ker_dxe_t, ker_dye_t, ker_dze_t, ker_dxh_t, ker_dyh_t, ker_dzh_t

#% This method returns the numerical error in loss calculation due to Finite Difference
    def Background_field_error(self, pad = 4):

        UI_exe          = self.Physical_attributes.Phase_modulation()
        ker_dxe_t, ker_dye_t, ker_dze_t, ker_dxh_t, ker_dyh_t, ker_dzh_t = self.Kernels([5,5,5,1,1])
        
        Looss1_r_UI     = tf.nn.conv3d(tf.math.real(UI_exe), ker_dze_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_i_UI     = tf.nn.conv3d(tf.math.imag(UI_exe), ker_dze_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_UI       = 1/(self.Physical_attributes.dx)*tf.complex(Looss1_r_UI,Looss1_i_UI)
        Looss1_r_UI2    = tf.nn.conv3d(tf.math.real(Looss1_UI), ker_dzh_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_i_UI2    = tf.nn.conv3d(tf.math.imag(Looss1_UI), ker_dzh_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss_UIz       = 1/(self.Physical_attributes.dx)*tf.complex(Looss1_r_UI2,Looss1_i_UI2)

        Error_UI        = Looss_UIz+((self.Physical_attributes.k)**2)*UI_exe
        
        mask            = np.ones((1, self.Physical_attributes.Ny, self.Physical_attributes.Nx, \
                                   self.Physical_attributes.Nz, 1), dtype = np.complex64)
        mask[:,:,:,0:pad,:] = 0
        mask[:,:,:,self.Physical_attributes.Nz-pad:self.Physical_attributes.Nz,:] = 0    
        mask            = tf.convert_to_tensor(mask)    
        Error_UI        = tf.multiply(Error_UI, mask)
        
        return Error_UI
        
#%  This is the main function which gets an input refractive index (RI), a 3D field (U_total)
# and a physical_attribute class and calculates the Hemholtz loss function
    def Maxwell_loss(self, RI_input, U_total):
        Urel, Uimag     = tf.split(U_total, 2, axis=4)
        # RI              = tf.cast(RI, dtype=tf.complex64)
        epsilonr        = (RI_input+self.Physical_attributes.n0)**2
        UI              = self.Physical_attributes.Phase_modulation()
        Error_UIt       = self.Background_field_error()
        ker_dxe_t, ker_dye_t, ker_dze_t, ker_dxh_t, ker_dyh_t, ker_dzh_t = self.Kernels([5,5,5,1,1])
        coeffx, coeffy, coeffz = self.Physical_attributes.PML()
        
          
        ## Field Envelope Define
        Feileds_, Feileds_rel, Feileds_img = self.complex_multiplication(tf.complex(Urel, Uimag), UI)
          
        ## Del2 Part Loss Calculation
        #X
        Looss1_r_v1 = tf.nn.conv3d(Feileds_rel, ker_dxe_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_i_v1 = tf.nn.conv3d(Feileds_img, ker_dxe_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_v1, Looss1_v1r, Looss1_v1i = self.complex_multiplication(coeffx,tf.complex(Looss1_r_v1,Looss1_i_v1))
         
        Looss1_r_v2 = tf.nn.conv3d(Looss1_v1r, ker_dxh_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_i_v2 = tf.nn.conv3d(Looss1_v1i, ker_dxh_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_v2   = 1/(self.Physical_attributes.dx**2)*self.complex_multiplication(coeffx,tf.complex(Looss1_r_v2,Looss1_i_v2), return_parts=False)
          
        #Y
        Looss1_r_v1 = tf.nn.conv3d(Feileds_rel, ker_dye_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_i_v1 = tf.nn.conv3d(Feileds_img, ker_dye_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss2_v1, Looss2_v1r, Looss2_v1i = self.complex_multiplication(coeffy,tf.complex(Looss1_r_v1,Looss1_i_v1))
         
        Looss1_r_v2 = tf.nn.conv3d(Looss2_v1r, ker_dyh_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_i_v2 = tf.nn.conv3d(Looss2_v1i, ker_dyh_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss2_v2   = 1/(self.Physical_attributes.dx**2)*self.complex_multiplication(coeffy,tf.complex(Looss1_r_v2,Looss1_i_v2), return_parts=False)
          
        #Z
        Looss1_r_v1 = tf.nn.conv3d(Feileds_rel, ker_dze_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_i_v1 = tf.nn.conv3d(Feileds_img, ker_dze_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss3_v1, Looss3_v1r, Looss3_v1i   = self.complex_multiplication(coeffz,tf.complex(Looss1_r_v1,Looss1_i_v1))
          
        Looss1_r_v2 = tf.nn.conv3d(Looss3_v1r, ker_dzh_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss1_i_v2 = tf.nn.conv3d(Looss3_v1i, ker_dzh_t, strides = (1,1,1,1,1), padding = 'SAME')
        Looss3_v2   = 1/(self.Physical_attributes.dx**2)*self.complex_multiplication(coeffz,tf.complex(Looss1_r_v2,Looss1_i_v2), return_parts=False)
          
        Looss_Us = Looss1_v2+Looss2_v2+Looss3_v2  #  This is the final loss of Del2 part
          
        ## Loss index Calculation 
        Looss_epsilon =  (self.Physical_attributes.k0**2)*(epsilonr)*(Feileds_)
        Looss4 = ((self.Physical_attributes.k0**2)*(RI_input)*(2.0*self.Physical_attributes.n0+RI_input))*(UI)
        Looss = Looss_Us + Looss_epsilon + Looss4 + Error_UIt
        Loss =  tf.reduce_mean((tf.math.abs(Looss))**2, axis = [0,1,2,3,4])
          
        return Loss