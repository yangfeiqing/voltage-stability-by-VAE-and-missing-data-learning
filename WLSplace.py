import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio


d=sio.loadmat('Ybus.mat')
Ybus=d['Ybus14']
nodes=Ybus.shape[0]

W=np.ones([nodes,1])
V=100
max_it=1000
LR=0.000001

x=tf.Variable(tf.random_normal([nodes,1]))

indii=[]
f_est=[]
for i in range(nodes):
    indi=np.nonzero(Ybus[i])
    indii.append(indi)
    indi_num=indi[0].shape[0]
    fi=1
    for j in range(indi_num):
        fi=fi*(1-x[indi[0][j]])
    f_est.append(fi)
f_est=tf.reshape(f_est,[nodes,1])

loss=tf.reduce_sum(tf.square(x)+V*tf.square(f_est))
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer()) 

losses=[]
for epoch in range(1000):
    
    losse=[]
    for num in range(2000):

        feed_dict = {}
        _, _loss, xx, ff= sess.run([train_op, loss, x, f_est], feed_dict) 
        losse.append(_loss)

    losses.append(np.mean(losse))
    print(epoch,100*losses[-1])





        
        
        
        
        
        
        
        
        