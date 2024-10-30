# -*- coding: utf-8 -*-
"""
Created on Tue Oct  8 13:46:25 2024

@author: KD
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


import torch
import matplotlib.pyplot as plt

x=torch.tensor([-1.,0.,1.,2.,3.,4.],dtype=torch.float32)
y=torch.tensor([-1.,1.,3.,5.,7.,9.])
w=torch.tensor(0.2,dtype=torch.float32,requires_grad=True)
b=torch.tensor(0.1,dtype=torch.float32,requires_grad=True)
def forward(x):
    return w*x+b
def loss(y,y_pred):
    return ((y_pred-y)**2).mean()

n_iter=1000
learning_rate=0.01
l_vec=[]
it=[]

for epoch in range(n_iter):
    y_pred=forward(x)       #pass-forward
    l=loss(y,y_pred)        #calc loss function
    l_vec.append(l.tolist())
    it.append(epoch)
    print("l=",l)
    l.backward()            #gradient
    print("Grad",w.grad)
    with torch.no_grad():
        w-=learning_rate*w.grad
        b-=learning_rate*b.grad
    w.grad.zero_()
    b.grad.zero_()
    
plt.plot(l_vec)
print(w,b)

#sprawdzenie graficzne
y_mod = []
for item in x:
    y_mod.append(w.detach().numpy()*item.numpy()+b.detach().numpy())

plt.figure()
plt.plot(x,y,'*')
plt.plot(x,y_mod)








