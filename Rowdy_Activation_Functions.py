#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 13 16:55:23 2021

@author: Dr. Ameya D. Jagtap, Assistant Professor of Applied Mathematics (Research), Brown University, USA.

#############################################################
    Rowdy Activation Functions for Function Approximation
#############################################################

References: 
    
1. Jagtap, Ameya D., Yeonjong Shin, Kenji Kawaguchi, and George Em Karniadakis,
Deep Kronecker neural networks: A general framework for neural networks with adaptive activation functions,
Neurocomputing 468 (2022): 165-180.

2. Jagtap, Ameya D., Kenji Kawaguchi, and George Em Karniadakis,
Adaptive activation functions accelerate convergence in deep and physics-informed neural networks,
Journal of Computational Physics 404 (2020): 109136.

3. Jagtap, Ameya D., Kenji Kawaguchi, and George Em Karniadakis, 
Locally adaptive activation functions with slope recovery for deep and physics-informed neural networks,
Proceedings of the Royal Society A 476, no. 2239 (2020): 20200334.

"""

#%%
import os
os.environ['CUDA_VISIBLE_DEVICES']='-1'
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
#from scipy.interpolate import griddata
import time
np.random.seed(1234)
tf.random.set_random_seed(1234)


def fun_x(x):
    
    f = np.zeros(len(x))
    f = np.reshape(f, (-1, 1))

    for i in range(len(x)):
                  
        f[i] = np.sin(200*np.pi*x[i]) 
        
    return f


def hyper_parameters_A(size): 
    a = tf.Variable(tf.constant(0.1, shape=size))

    return a

def hyper_parameters_freq1(size):
    return tf.Variable(tf.constant(0.1, shape=size))

def hyper_parameters_freq2(size):
    return tf.Variable(tf.constant(0.1, shape=size))

def hyper_parameters_freq3(size):
    return tf.Variable(tf.constant(0.1, shape=size))

def hyper_parameters_amplitude(size):
    return tf.Variable(tf.constant(0.0, shape=size))

def hyper_parameters(size):
    return tf.Variable(tf.random_normal(shape=size, mean = 0.0, stddev = 0.1))

def DNN(X, W, b,a, a1, a2, a3, a4, a5, a6, a7, a8, a9, F1, F2, F3, F4, F5, F6, F7, F8, F9):
    A = X
    L = len(W)
    for i in range(L - 1):
        A =  tf.cos(10*a[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a1[i]*tf.sin(10*F1[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a2[i]*tf.sin(20*F2[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a3[i]*tf.sin(30*F3[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a4[i]*tf.sin(40*F4[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a5[i]*tf.sin(50*F5[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a6[i]*tf.sin(60*F6[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a7[i]*tf.sin(70*F7[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a8[i]*tf.sin(80*F8[i]*tf.add(tf.matmul(A, W[i]), b[i]))\
            + 10*a9[i]*tf.sin(90*F9[i]*tf.add(tf.matmul(A, W[i]), b[i]))
    Y = tf.add(tf.matmul(A, W[-1]), b[-1])
    return Y

if __name__ == "__main__":
   
    # Number of training data points
    N = 100
    x = np.linspace(-np.pi,np.pi, N+1)
    x = np.reshape(x, (-1, 1))

    # High frequency sine function
    y = fun_x(x)

    # 
    layers = [1] + 3*[50] + [1]
    
    # Initialize the trainable parameters
    W = [hyper_parameters([layers[l-1], layers[l]]) for l in range(1, len(layers))]
    b = [hyper_parameters([1, layers[l]]) for l in range(1, len(layers))]

    a  = [hyper_parameters_A([1]) for l in range(1, len(layers))]
    a1 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    a2 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    a3 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    a4 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    a5 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    a6 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    a7 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    a8 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    a9 = [hyper_parameters_amplitude([1]) for l in range(1, len(layers))]
    
    F1 = [hyper_parameters_freq1([1]) for l in range(1, len(layers))]
    F2 = [hyper_parameters_freq2([1]) for l in range(1, len(layers))]
    F3 = [hyper_parameters_freq3([1]) for l in range(1, len(layers))]
    F4 = [hyper_parameters_freq1([1]) for l in range(1, len(layers))]
    F5 = [hyper_parameters_freq2([1]) for l in range(1, len(layers))]
    F6 = [hyper_parameters_freq3([1]) for l in range(1, len(layers))] 
    F7 = [hyper_parameters_freq1([1]) for l in range(1, len(layers))]
    F8 = [hyper_parameters_freq2([1]) for l in range(1, len(layers))]
    F9 = [hyper_parameters_freq3([1]) for l in range(1, len(layers))] 

    
    x_train = tf.placeholder(tf.float32, shape=[None, 1])
    y_train = tf.placeholder(tf.float32, shape=[None, 1])

    # Training
    start_time = time.time() 
    y_pred = DNN(x_train, W, b,a, a1, a2, a3, a4, a5, a6, a7, a8, a9, F1, F2, F3, F4, F5, F6, F7, F8, F9)

    loss = tf.reduce_mean(tf.square(y_pred - y_train)) 
    train = tf.train.AdamOptimizer(4e-6).minimize(loss)


    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    nmax = 20001
    n = 0
    err = 1.0
    MSE_hist = []
    l2_err = []
    Sol = []

    while n <= nmax: 
        n = n + 1
        
        if n %1 == 0:
            loss_, _, y_ = sess.run([loss, train, y_pred], feed_dict={x_train: x, y_train: y})
            err = loss_
            l2_error = np.linalg.norm(y-y_,2)/np.linalg.norm(y,2)
            MSE_hist.append(err)
            l2_err.append(l2_error)
        
        if n == 5000 or n==10000 or n==15000: 
            Sol.append(y_)
            print('Steps: %d, loss: %.3e'%(n, loss_))  
            
    Solution = np.concatenate(Sol, axis=1)
    elapsed = time.time() - start_time                
    print('Training time: %.4f' % (elapsed))
     
    ############################################ PLOTTING

    fig = plt.figure(1)
    plt.plot(MSE_hist, 'b-', linewidth = 1)
    plt.xlabel('$\#$ iterations', fontsize = 35)
    plt.ylabel('Loss', fontsize = 35)
    plt.yscale('log')
    plt.tick_params(axis="x", labelsize = 20)
    plt.tick_params(axis="y", labelsize = 20)  
    plt.grid()
    fig.set_size_inches(w=9.6,h=8)
    plt.savefig('Sine_MSEhist.pdf') 
    #%
    
    fig = plt.figure(2)
    plt.plot(l2_err, 'b-')
    plt.xlabel('$\#$ iterations', fontsize = 35)
    plt.ylabel('Rel. $L_2$ error', fontsize = 35)
    plt.yscale('log')
    plt.tick_params(axis="x", labelsize = 20)
    plt.tick_params(axis="y", labelsize = 20)  
    plt.grid()
    fig.set_size_inches(w=9.6,h=8)
    plt.savefig('Sine_L2hist.pdf') 

