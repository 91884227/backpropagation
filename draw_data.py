#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[22]:


import numpy as np
import matplotlib.pyplot as plt
import json


# # function here

# In[19]:


def show_ground_true(x, y):
    plt.title("Ground truth", fontsize = 18)
    for i in range(x.shape[0]):
        if( y[i] == 0):
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")    


# In[20]:


def show_result(x, y, pred_y):
    plt.subplot(1, 2, 1)
    plt.title("Ground truth", fontsize = 18)
    for i in range(x.shape[0]):
        if( y[i] == 0):
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")
    
    plt.subplot(1, 2, 2)
    plt.title("Predict result", fontsize = 18)
    for i in range(x.shape[0]):
        if pred_y[i]==0:
            plt.plot(x[i][0], x[i][1], "ro")
        else:
            plt.plot(x[i][0], x[i][1], "bo")
    plt.show()


# In[44]:


if __name__ == "__main__":
    print("print XOR data")
    temp2 = "./data/%s.npy"
    X = np.load(temp2 % "X_train_XOR", allow_pickle = True)
    y = np.load(temp2 % "y_train_XOR", allow_pickle = True)
    show_ground_true(X, y)


# In[ ]:




