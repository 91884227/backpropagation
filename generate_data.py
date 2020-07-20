#!/usr/bin/env python
# coding: utf-8

# # import tool

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import json


# In[2]:


def generate_XOR_easy():
    inputs, labels = [ ] , [ ]
    
    for i in range(11):
        inputs.append([0.1*i, 0.1*i])
        labels.append(0)
        
        if 0.1*i == 0.5:
            continue
        
        inputs.append([0.1*i, 1 - 0.1*i])
        labels.append(1)
    
    return np.array(inputs), np.array(labels).reshape(21, 1)


# In[3]:


def generate_linear(n = 100):
    pts = np.random.uniform(0, 1, (n, 2))
    inputs, labels = [ ] , [ ]
    for pt in pts:
        inputs.append([pt[0], pt[1]])
        distance = (pt[0] - pt[1])/1.414
        if pt[0] > pt[1]:
            labels.append(0)
        else:
            labels.append(1)
    return np.array(inputs), np.array(labels).reshape(n, 1)


# In[17]:


if __name__ == "__main__":
    X, y = generate_linear(n = 100)
    X_train = X#.tolist()
    y_train = y#.tolist()
    
    temp = "./data/%s"
#     with open(temp % "X_train_linear", 'w') as outfile:
#         json.dump(X_train, outfile)
#     with open(temp % "y_train_linear", 'w') as outfile:
#         json.dump(y_train, outfile)    
        
    np.save(temp % "X_train_linear", X_train)
    np.save(temp % "y_train_linear", y_train)

    X, y = generate_XOR_easy()
    X_train = X#.tolist()
    y_train = y#.tolist()
    
    temp = "./data/%s"
#     with open(temp % "X_train_XOR", 'w') as outfile:
#         json.dump(X_train, outfile)
#     with open(temp % "y_train_XOR", 'w') as outfile:
#         json.dump(y_train, outfile)    

    np.save(temp % "X_train_XOR", X_train)
    np.save(temp % "y_train_XOR", y_train)