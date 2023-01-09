#!/usr/bin/env python
import tensorflow as tf
from tqdm import tqdm, trange
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

def plot_history(history,lr,opt):
  dim = np.arange(1,len(history[0]['loss'])+1,1)
  # df_hist = pd.DataFrame(history,index=dim)
  # df_hist.to_csv(os.path.join('LAB3',f'history-gradient descent LR{lr}.csv'))
  for i in range(len(lr)):
    plt.plot(dim,history[i]['loss'],label=f'optimizer:{opt[i]}')
  
  plt.title(f'Loss')
  plt.ylabel('Loss')
  plt.xlabel('Iteration')
  plt.yscale('log')

  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig(os.path.join('LAB3',f'history-gradient descent_opt.jpg'))
  plt.close()

def GD(optimizer,lr=0.01):
  # Model parameters
  W = tf.Variable([.3], dtype=tf.float32)
  b = tf.Variable([-.3], dtype=tf.float32)
  # Model input and output
  x = tf.placeholder(tf.float32)
  linear_model = W * x + b
  y = tf.placeholder(tf.float32)

  # loss
  loss = tf.reduce_sum(tf.square(linear_model - y)) # sum of the squares
  # optimizer
  # optimizer = tf.train.GradientDescentOptimizer(lr)
  # optimizer = optimizer
  train = optimizer.minimize(loss)

  # training data
  x_train = [1, 2, 3, 4]
  y_train = [0, -1, -2, -3]
  # training loop
  init = tf.global_variables_initializer()
  sess = tf.Session()
  sess.run(init) # reset values to wrong

  curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
  print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

  pbar = tqdm(total=1000, ncols=120)
  history = {'lr':lr,'w':[],'b':[],'loss':[]}
  for i in range(1000):
    sess.run(train, {x: x_train, y: y_train})
    curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_train, y: y_train})
    history['w'].append(curr_W)
    history['b'].append(curr_b)
    history['loss'].append(curr_loss)
    # print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    pbar.set_description("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))
    pbar.update(1)
  return history

LR = [0.01,0.001,0.0001]
hist_1 = GD(tf.train.GradientDescentOptimizer(LR[0]))
hist_2 = GD(tf.train.RMSPropOptimizer(LR[0]))
hist_3 = GD(tf.train.AdamOptimizer(LR[0]))

plot_history([hist_1,hist_2,hist_3],LR,['GD','RMSProp','Adam'])