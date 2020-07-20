#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import math


# In[2]:


class output_layer:
    def __init__(self, in_features_):
        k = 1/in_features_
        self.W = np.random.uniform(-math.sqrt(k), math.sqrt(k), (in_features_, 1))
        # self.W = np.array([[0.14], [0.15]])
        self.Gradient_Z = None
        self.Gradient_W = None
        self.y_pred = None
        self.y = None 
        self.Z = None
    
    def sigmoid(self,x):
        return( 1.0/(1.0+np.exp(-x)) )
    
    def derivative_sigmoid(self, x):
        return( np.multiply(self.sigmoid(x),
                            1.0 - self.sigmoid(x)
                           )
              )
    
    def forward(self, input_):
        self.input = input_
        self.Z = np.matmul(input_, self.W)
        self.y_pred = self.sigmoid(self.Z)
        return( self.y_pred) 
    
    def loss(self, y):
        self.y = y
        return( (y - self.y_pred)**2 )
        # print( (y - self.y_pred)**2 )
    
    def cal_Gradient_Z(self):
        self.Gradient_Z =  float( (self.y_pred - self.y)*self.derivative_sigmoid( self.Z ) )
        
    def cal_Gradient_W(self):
        self.Gradient_W = np.matmul(np.array(self.input).reshape(-1, 1), 
                                    np.array([[self.Gradient_Z]]) )  


# In[3]:


class Layer:
    def __init__(self, in_features_, out_feature, backlayer_):
        k = 1/in_features_
        self.W = np.random.uniform(-math.sqrt(k), math.sqrt(k), (in_features_, out_feature))
        # self.W = np.array([[0.11, 0.12], [0.21, 0.08]])
        self.Gradient_Z = None
        self.Gradient_W = None
        self.output = None 
        self.backlayer = backlayer_

    def sigmoid(self,x):
        return( 1.0/(1.0+np.exp(-x)) )
    
    def derivative_sigmoid(self, x):
        return( np.multiply(self.sigmoid(x),
                            1.0 - self.sigmoid(x)
                           )
              )
    
    def forward(self, input_):
        self.input = input_
        self.Z = np.matmul(self.input, self.W)
        self.output = self.sigmoid(self.Z)
        return( self.output ) 
    
    def cal_Gradient_Z(self):
        buf1 = np.matmul(self.backlayer.W, 
                         np.array(self.backlayer.Gradient_Z).reshape(-1, 1))
        
        buf2 = self.derivative_sigmoid(self.Z).reshape(-1, 1)
        self.Gradient_Z = np.multiply(buf1, buf2)
    
    def cal_Gradient_W(self):
        self.Gradient_W = np.matmul(np.array(self.input).reshape(-1, 1), 
                                    self.Gradient_Z.T)


# In[4]:


class net:
    def __init__(self, num_1 = 2, num_2 = 4, num_3 = 4, LR_ = 0.005):
        self.LR = LR_
        self.outlayer = output_layer(num_3)
        self.h_layer_2 = Layer(num_2, num_3, self.outlayer )
        self.h_layer_1 = Layer(num_1, num_2, self.h_layer_2)
    
    def forward(self, input_):
        buf = self.h_layer_1.forward(input_)
        buf = self.h_layer_2.forward(buf)
        buf = self.outlayer.forward(buf)  
    
    def loss(self, y):
        buf = self.outlayer.loss(y)
        return(buf)
    
    
    def cal_gradient(self):
        self.outlayer.cal_Gradient_Z()
        self.outlayer.cal_Gradient_W()
        
        self.h_layer_2.cal_Gradient_Z()
        self.h_layer_2.cal_Gradient_W()
        
        self.h_layer_1.cal_Gradient_Z()
        self.h_layer_1.cal_Gradient_W()
    
    def update(self):
        self.outlayer.W = self.outlayer.W - self.outlayer.Gradient_W * self.LR
        self.h_layer_2.W = self.h_layer_2.W - self.h_layer_2.Gradient_W * self.LR
        self.h_layer_1.W = self.h_layer_1.W - self.h_layer_1.Gradient_W * self.LR            


# In[5]:


if __name__ == "__main__":
    net1 = net(LR_ = 0.05)
    for i in range(20):
        net1.forward([2, 3])
        net1.loss(1)
        net1.cal_gradient()
        net1.update()
