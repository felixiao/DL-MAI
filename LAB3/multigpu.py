import time

import tensorflow as tf
# from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.python.client import device_lib
import numpy as np
import read_inputs

def check_available_gpus():
    local_devices = device_lib.list_local_devices()
    gpu_names = [x.name for x in local_devices if x.device_type == 'GPU']
    gpu_num = len(gpu_names)

    print('{0} GPUs are detected : {1}'.format(gpu_num, gpu_names))

    return gpu_num


def model(X, reuse=False):
    with tf.variable_scope('L1', reuse=reuse):
        L1 = tf.layers.conv2d(X, 32, [5, 5], reuse=reuse)
        L1 = tf.layers.max_pooling2d(L1, [2, 2], [2, 2])
        L1 = tf.layers.dropout(L1, 0.5, True)

    with tf.variable_scope('L2', reuse=reuse):
        L2 = tf.layers.conv2d(L1, 64, [5, 5], reuse=reuse)
        L2 = tf.layers.max_pooling2d(L2, [2, 2], [2, 2])
        L2 = tf.layers.dropout(L2, 0.5, True)

    with tf.variable_scope('L3', reuse=reuse):
        L3 = tf.contrib.layers.flatten(L2)
        L3 = tf.layers.dense(L3, 1024, activation=tf.nn.relu)
        L3 = tf.layers.dropout(L3, 0.5, True)

    with tf.variable_scope('LF', reuse=reuse):
        LF = tf.layers.dense(L3, 10, activation=None)

    return LF


if __name__ == '__main__':
    # need to change learning rates and batch size by number of GPU
    batch_size = 100
    learning_rate = 0.001
    total_epoch = 1

    gpu_num = check_available_gpus()

    X = tf.placeholder(tf.float32, [None, 28, 28, 1])
    Y = tf.placeholder(tf.float32, [None, 10])

    losses = []
    X_A = tf.split(X, int(gpu_num))
    Y_A = tf.split(Y, int(gpu_num))

    '''
    Multi GPUs Usage
    Results on P40
     * Single GPU computation time: 0:00:22.252533
     * 2 GPU computation time: 0:00:12.632623
     * 4 GPU computation time: 0:00:11.083071
     * 8 GPU computation time: 0:00:11.990167
     
    Need to change batch size and learning rates
         for training more efficiently
    
    Reference: https://research.fb.com/wp-content/uploads/2017/06/imagenet1kin1h5.pdf
    '''
    for gpu_id in range(int(gpu_num)):
        with tf.device(tf.DeviceSpec(device_type="GPU", device_index=gpu_id)):
            with tf.variable_scope(tf.get_variable_scope(), reuse=(gpu_id > 0)):
                cost = tf.nn.softmax_cross_entropy_with_logits(
                                logits=model(X_A[gpu_id], gpu_id > 0),
                                labels=Y_A[gpu_id])
                losses.append(cost)

    loss = tf.reduce_mean(tf.concat(losses, axis=0))

    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(
        loss, colocate_gradients_with_ops=True)  # Important!

    init = tf.global_variables_initializer()
    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)

    data_input,_,_,_ = read_inputs.load_data_mnist('LAB3/MNIST_data/mnist.pkl.gz')
    train_set_x, train_set_y = data_input[0]
    val_set_x, val_set_y     = data_input[1]
    test_set_x, test_set_y   = data_input[2]

    # convert to one-hot encoding 
    real_output = np.zeros( (len(train_set_y) , 10), dtype=np.float )
    for i in range ( len(train_set_y) ):
        real_output[i][train_set_y[i]] = 1.0  


    #data layout changes since output should an array of 10 with probabilities
    real_check = np.zeros( (len(test_set_y), 10), dtype=np.float )
    for i in range (len(test_set_y) ):
        real_check[i][test_set_y[i]] = 1.0

    total_batch = int(len(train_set_x)/batch_size)
    print("total: %s,step: %s,batch size: %s" % (len(train_set_x), total_batch, batch_size))

    start_time = time.time()

    for epoch in range(total_epoch):
        total_cost = 0

        for i in range(total_batch):
            batch_xs = train_set_x[i*batch_size:(i+1)*batch_size]
            batch_ys = real_output[i*batch_size:(i+1)*batch_size]
            batch_xs = batch_xs.reshape(-1, 28, 28, 1)
            _, cost_val = sess.run([optimizer, loss],
                                   feed_dict={X: batch_xs,
                                              Y: batch_ys})
            total_cost += cost_val

        # print("total cost : %s" % total_cost)

    print("--- Training time : {0} seconds /w {1} GPUs ---".format(
        time.time()- start_time, gpu_num))
    
    res = np.mean([sess.run([loss], feed_dict={ X: test_set_x[i*batch_size:(i+1)*batch_size],
                                        Y: real_check[i*batch_size:(i+1)*batch_size]}) for i in range(total_batch)])
    print('test accuracy %.3f' %(res))