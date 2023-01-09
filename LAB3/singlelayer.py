#!/usr/bin/env python
import tensorflow as tf
import read_inputs
import numpy as N
import matplotlib.pyplot as plt
import os
import numpy as np

def plot_mnist(image,label,id,split):
  plt.imshow(image,cmap='gray')
  plt.savefig(os.path.join('LAB3',f'mnist_singlelayer_{split}_{id}:{label}.jpg'))
  plt.close()

def plot_history(history,lr,opt):
  dim = np.arange(1,len(history[0]['loss'])+1,1)
  # df_hist = pd.DataFrame(history,index=dim)
  # df_hist.to_csv(os.path.join('LAB3',f'history-gradient descent LR{lr}.csv'))
  for i in range(len(lr)):
    plt.plot(dim,history[i]['loss'],label=f'optimizer:{opt[i]}@lr:{lr[i]}')
  
  plt.title(f'Loss')
  plt.ylabel('Loss')
  plt.xlabel('Step')

  plt.legend(loc='upper right')
  plt.tight_layout()
  plt.savefig(os.path.join('LAB3',f'history-singlelayer_opt.jpg'))
  plt.close()

#read data from file
data_input = read_inputs.load_data_mnist('LAB3/MNIST_data/mnist.pkl.gz')
#FYI data = [(train_set_x, train_set_y), (valid_set_x, valid_set_y), (test_set_x, test_set_y)]
data = data_input[0]

# data : (datasplit[train,val, test] =3 , (image,label) =2, (image/label len)=50000, (image/label size)=768/10)
# data[0][0]: the all train images
# data[0][1]: the all train labels
# data[0][0][0] the first train image 1/50000, size = 768
# for i in range(10):
#   image = data[1][0][i].reshape((28,28))
#   label = data[1][1][i]
#   plot_mnist(image,label,i,'val')

#data layout changes since output should an array of 10 with probabilities
real_output = N.zeros( (N.shape(data[0][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[0][1])[0] ): 
  real_output[i][data[0][1][i]] = 1.0  

#data layout changes since output should an array of 10 with probabilities
real_check = N.zeros( (N.shape(data[2][1])[0] , 10), dtype=N.float )
for i in range ( N.shape(data[2][1])[0] ):
  real_check[i][data[2][1][i]] = 1.0

def train(optimizer,lr=0.5):
  #set up the computation. Definition of the variables.
  x = tf.placeholder(tf.float32, [None, 784])
  W = tf.Variable(tf.zeros([784, 10]))
  b = tf.Variable(tf.zeros([10]))
  y = tf.nn.softmax(tf.matmul(x, W) + b)
  y_ = tf.placeholder(tf.float32, [None, 10])

  cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
  train_step = optimizer(lr).minimize(cross_entropy)

  sess = tf.InteractiveSession()
  tf.global_variables_initializer().run()

  #TRAINING PHASE
  print("TRAINING")
  history={'loss':[]}
  # batch size = 100; step = 500
  for i in range(500):
    batch_xs = data[0][0][100*i:100*i+100]
    batch_ys = real_output[100*i:100*i+100]
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    history['loss'].append(sess.run([cross_entropy],{x: batch_xs, y_: batch_ys}))

  #CHECKING THE ERROR
  print("ERROR CHECK")

  correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
  accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
  test_loss=sess.run(accuracy, feed_dict={x: data[2][0], y_: real_check})

  return history, test_loss

history1,t_loss1 = train(tf.train.GradientDescentOptimizer,lr=0.5)
history2,t_loss2 = train(tf.train.RMSPropOptimizer,lr=0.01)
history3,t_loss3 = train(tf.train.AdamOptimizer,lr=0.01)
plot_history([history1,history2,history3],[0.5,0.01,0.01],['GD','RMSProp','Adam'])
print('GD test loss:',t_loss1)
print('RMSProp test loss:',t_loss2)
print('Adam test loss:',t_loss3)