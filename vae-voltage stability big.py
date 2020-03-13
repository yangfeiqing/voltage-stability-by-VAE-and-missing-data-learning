# -*- coding: utf-8 -*-
"""
Created on Fri Jun  8 15:00:51 2018

@author: sjtu
"""

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.contrib.distributions as tfd
import scipy.io as sio

d=sio.loadmat('DNNtrain14_1.mat')

all_VM=d['VM']
all_VA=d['VA']
all_VP=d['VP']
all_VQ=d['VQ']

nodes=2*all_VM.shape[0]
#p=d['p']
#nose=d['nose'][0][0]

#cla=np.zeros(p.shape[1])
#cla=np.around(3*p/np.max(p))
#for t in range(nose,p.shape[1]):
#    cla[0,t]=4
#cla=cla.transpose().reshape(p.shape[1])

train_VM=(all_VM-np.min(all_VM))/(np.max(all_VM)-np.min(all_VM))
train_VA=(all_VA-np.min(all_VA))/(np.max(all_VA)-np.min(all_VA))
train_y=np.concatenate((train_VM,train_VA),axis=0)
train_y=train_y.transpose()
train_yall=train_y
train_y=train_y[0:2000,:]

dt=sio.loadmat('DNNtest14_1.mat')

test_VM=dt['VM']
test_VA=dt['VA']
test_VP=dt['VP']
test_VQ=dt['VQ']

test_VM=(test_VM-np.min(all_VM))/(np.max(all_VM)-np.min(all_VM))
test_VA=(test_VA-np.min(all_VA))/(np.max(all_VA)-np.min(all_VA))
test_y=np.concatenate((test_VM,test_VA),axis=0)
test_y=test_y.transpose()


dt2=sio.loadmat('DNNtrain14_2.mat')

test2_VM=dt2['VM']
test2_VA=dt2['VA']
test2_VP=dt2['VP']
test2_VQ=dt2['VQ']

test2_VM=(test2_VM-np.min(all_VM))/(np.max(all_VM)-np.min(all_VM))
test2_VA=(test2_VA-np.min(all_VA))/(np.max(all_VA)-np.min(all_VA))
test2_y=np.concatenate((test2_VM,test2_VA),axis=0)
test2_y=test2_y.transpose()
#cla=cla[0:int(0.9*train_y.shape[0])]
#train_y=train_y[0:int(0.9*train_y.shape[0])]
hidden=300


def make_encoder(data, code_size):
    
    x = tf.layers.flatten(data)
    x = tf.layers.dense(x, hidden, tf.nn.relu)
    x = tf.layers.dense(x, hidden, tf.nn.relu)
    loc = tf.layers.dense(x, code_size)
    scale = tf.layers.dense(x, code_size, tf.nn.softplus)
    
    return tfd.MultivariateNormalDiag(loc, scale), loc, scale

def make_prior(code_size):
#    produce guassian N(0,1)
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    
    return tfd.MultivariateNormalDiag(loc, scale)

def make_decoder(code, data_shape):
    
    x = code
    x = tf.layers.dense(x, hidden, tf.nn.relu)
    x = tf.layers.dense(x, hidden, tf.nn.relu)
    logit = tf.layers.dense(x, np.prod(data_shape))
    logit = tf.reshape(logit, [-1] + data_shape)
    
    return tfd.Independent(tfd.Bernoulli(logit), 2)


make_encoder = tf.make_template('encoder', make_encoder)
make_decoder = tf.make_template('decoder', make_decoder)

data = tf.placeholder(tf.float32, [None, nodes, 1])

prior = make_prior(code_size=2)
posterior, loc, scale = make_encoder(data, code_size=2)
code_all = posterior.sample(10)
code = tf.reduce_mean(code_all,reduction_indices=0)

likelihood = make_decoder(code, [nodes, 1]).log_prob(data)
divergence = tfd.kl_divergence(posterior, prior)
elbo = tf.reduce_mean(likelihood - divergence)

optimize = tf.train.AdamOptimizer(0.0001).minimize(-elbo)
samples = make_decoder(prior.sample(10), [nodes, 1]).mean()

init = tf.global_variables_initializer()  
sess = tf.Session()
sess.run(init)

if __name__ == '__main__':
    
    for epoch in range(20):
        
        test_elbo, test_codes, test_samples, mean, var, alll = sess.run(
                [elbo, code, samples, loc, scale, code_all], {data: train_y.reshape(train_y.shape[0],nodes,1)})
        print('Epoch', epoch, 'elbo', test_elbo)
        plt.scatter(mean[:, 0], mean[:, 1], s=1)
        plt.axis([-0.1,0.35,-1,0.6])
        plt.xlabel('z1');plt.ylabel('z2')
#        plt.colorbar()  
        plt.show()         
#        plt.plot(mean[:, 0], mean[:, 1])
        
#    plot_codes(test_codes)
#    plot_sample(test_samples)
        for _ in range(1000):
            sess.run(optimize, {data: train_y.reshape(train_y.shape[0],nodes,1)})
            
#    num=int(0.9*train_y.shape[0])  
#    numfull= train_y.shape[0]       
#    step=np.linspace(1,nose, nose,dtype=np.float32)
#    steps=np.linspace(1,numfull, numfull,dtype=np.float32)
#    VM_nose=all_VM.transpose()
#    plt.plot(p[0,0:numfull], all_VM[3,0:numfull].flatten(), 'r-',label='PV curve')
#    plt.plot(p_plot[0:217], dis_chanor[0:217].flatten(), 'b-',label='autoencoder index2')
#    plt.legend(loc='upper right')
#    plt.xlabel('P/MW');plt.ylabel('index')
    plt.draw();plt.ioff(); plt.show()
    test_elbo, test_codes, test_samples, mean, var, alll = sess.run(
            [elbo, code, samples, loc, scale, code_all], {data: test_y.reshape(test_y.shape[0],28,1)})    
    
    
    
    
#    
#    px=p[0,0:numfull]
#    py=all_VM[3,0:numfull]
#
#    pxmax=np.max(px)
#    pxmin=np.min(px)
#    pymax=np.max(py)
#    pymin=np.min(py)
#    barx=(mean[:, 0]-np.min(mean[:, 0]))/(np.max(mean[:, 0])-np.min(mean[:, 0]))*(pxmax-pxmin)+pxmin
#    bary=(mean[:, 1]-np.min(mean[:, 1]))/(np.max(mean[:, 1])-np.min(mean[:, 1]))*(pymax-pymin)+pymin
#    plt.plot(px, py.flatten(), 'r-',label='PV curve')
#    plt.scatter(barx, bary)      
##    plt.legend(loc='upper right')
##    plt.xlabel('Load Demand');plt.ylabel('Voltage Magnitude')
#    plt.draw();plt.ioff(); plt.show()   