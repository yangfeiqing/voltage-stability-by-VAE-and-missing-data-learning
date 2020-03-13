

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.distributions as tfd
import scipy.io as sio
from tensorflow.examples.tutorials.mnist import input_data
import random

d=sio.loadmat('DNNfull57_2.mat')


all_VM=d['VM']
all_VA=d['VA']
all_VP=d['VP']
all_VQ=d['VQ']
p=d['lamall']
leng=d['leng']
nose=d['nose']
nodes=2*all_VM.shape[0]

lengall=np.zeros(leng.shape)
for i in range(leng.shape[1]):
    lengall[0][i]=np.sum(leng[0][0:i+1])

noseall=np.zeros(nose.shape)
for i in range(leng.shape[1]):
    noseall[0][i]=np.sum(leng[0][0:i])+nose[0][i]
    
ps=np.zeros(p.shape)
for i in range(lengall.shape[1]):
    if i==0:
        pss=p[0][0:int(lengall[0][i])]
        pss=(pss-np.min(pss))/(np.max(pss)-np.min(pss))
        ps[0][0:int(lengall[0][i])]=pss
    else:            
        pss=p[0][int(lengall[0][i-1]):int(lengall[0][i])]
        pss=(pss-np.min(pss))/(np.max(pss)-np.min(pss))
        ps[0][int(lengall[0][i-1]):int(lengall[0][i])]=pss

winth=5
def sample(winth):
    a=int(random.random()*(leng.shape[1]))
    if a==0:
        b=int(random.random()*(lengall[a]-winth))
    b=int(random.random()*(lengall[0][a]-lengall[0][a-1]-winth)+lengall[0][a-1])
    data_use_VM=all_VM[:,b:(b+winth)]
    data_use_VA=all_VA[:,b:(b+winth)]
    data_use=np.concatenate([data_use_VM,data_use_VA],axis=0)
    
    return data_use

ad=sample(5)


def add_layer(inputs, in_size1, in_size2, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size2, out_size]))
    biases = tf.Variable(tf.zeros([in_size1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

in_size1=winth
in_size2=nodes 
mid1=2*winth
mid2=2*winth
fea_len=8
LR=0.002

#tf.reset_default_graph()
xs=tf.placeholder(tf.float32, [in_size1,in_size2])  
ys=tf.placeholder(tf.float32, [1, 1]) 

x_h1=add_layer(xs, in_size1, in_size2, mid1,tf.nn.sigmoid)
x_h2=add_layer(x_h1, in_size1, mid1, mid2,tf.nn.sigmoid)
x_h3=add_layer(x_h1, in_size1, mid2, fea_len,tf.nn.sigmoid)
feature4=tf.reduce_max(x_h3, reduction_indices=[0])
feature4=tf.reshape(feature4,[1,fea_len])
x_h5=tf.layers.dense(feature4,winth,activation=tf.nn.sigmoid)
y=tf.layers.dense(x_h5,1,activation=tf.nn.sigmoid)

outs = tf.reshape(y, [1,1])          # reshape back to 3D

loss = tf.losses.mean_squared_error(labels=ys, predictions=outs)  # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 
