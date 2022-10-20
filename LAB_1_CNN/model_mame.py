from distutils.spawn import find_executable
from multiprocessing.dummy import Pool
import pandas as pd
import numpy as np
import os
os.environ['PYTHONHASHSEED']=str(123)

import matplotlib.pyplot as plt
import time
import glob
import shutil
import json
import random
import tensorflow as tf
#Define the NN architecture
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Rescaling
from keras.losses import CategoricalCrossentropy
from keras.utils.vis_utils import plot_model


class Resnet_Block(tf.keras.Model):
    def __init__(self, filters=(32,32),kernel_sizes=(3,3)):
        super(Resnet_Block, self).__init__(name='')
        filters1, filters2 = filters
        size1, size2 = kernel_sizes

        self.conv2a = tf.keras.layers.Conv2D(filters1, size1, padding='same')
        self.bn2a = tf.keras.layers.BatchNormalization()

        self.conv2b = tf.keras.layers.Conv2D(filters2, size2, padding='same')
        self.bn2b = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=False):
        x = self.conv2a(input_tensor)
        x = self.bn2a(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2b(x)
        x = self.bn2b(x, training=training)

        x += input_tensor
        return  tf.nn.relu(x)


class MAMe_CNN():
    def __init__(self, param):
        self.param = param
        os.environ['PYTHONHASHSEED']=str(123)
        tf.random.set_seed(param.seed)
        np.random.seed(param.seed)
        random.seed(param.seed)

    def cnn_block(self,x, channels=32,BN=True,activation='relu', Pooling='Max',kernel_size=3, cn_strides=1,cn_padding = 'valid',pool_size=2,pool_strides=2,pool_padding='valid'):
        x = tf.keras.layers.Conv2D(channels, kernel_size=kernel_size, strides=cn_strides, padding=cn_padding)(x)
        if BN:
            x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation(activation)(x)
        if Pooling == 'Max':
            x = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)(x)
        elif Pooling == 'Avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)(x)
        return x
    
    def input_block(self,x,agu_flip=True,agu_rot=True):
        x = tf.keras.layers.Rescaling(1./255)(x)
        # x = tf.keras.layers.ZeroPadding2D((3, 3))(x)
        if agu_flip:
            x = tf.keras.layers.RandomFlip(seed=self.param.seed)(x)
        if agu_rot:
            x = tf.keras.layers.RandomRotation(seed=self.param.seed,factor=0.2, fill_mode='reflect')(x)
        return x

    def output_block(self,x,AvgPool=False,Flatten=False,GAP=True,FC=None,Activation=tf.nn.softmax):
        if AvgPool:
            x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
        if Flatten:
            x = tf.keras.layers.Flatten()(x)
        if GAP:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if FC:
            for s in FC:
                x = tf.keras.layers.Dense(s, activation = 'relu')(x)
        x = tf.keras.layers.Dense(self.param.num_classes, activation = Activation)(x)
        return x

    def identity_block(self,x, filters=[16,32],kernel_size=3):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filters[0], kernel_size, padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filters[1], kernel_size, padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    def maxpool_block(self,x,filters=[16,32],kernel_size=3):
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filters[0], kernel_size, padding = 'same', strides = (2,2))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filters[1], kernel_size, padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Downsampling with MaxPool2D
        x = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(x)

        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        return x

    def convolutional_block(self,x, filters=[16,32],kernel_size=3):
        # copy tensor to variable called x_skip
        x_skip = x
        # Layer 1
        x = tf.keras.layers.Conv2D(filters[0], kernel_size, padding = 'same', strides = (2,2))(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Layer 2
        x = tf.keras.layers.Conv2D(filters[1], kernel_size, padding = 'same')(x)
        x = tf.keras.layers.BatchNormalization(axis=3)(x)
        x = tf.keras.layers.Activation('relu')(x)
        # Processing Residue with conv(1,1)
        x_skip = tf.keras.layers.Conv2D(filters[1], 1, strides = (2,2))(x_skip)
        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)
        return x
    
    def Residue_Model(self):
        # Step 1 (Setup Input Layer)
        x_input = tf.keras.layers.Input(self.param.input_shape)
        
        # Layer 1
        x = self.input_block(x_input)

        # Layer 2 (Initial Conv layer along with maxPool)
        x = self.cnn_block(x,channels=64, kernel_size=7, cn_strides=1, cn_padding='same',
            BN=True,activation='relu',
            Pooling='Max',pool_size=2,pool_strides= 2,pool_padding = 'same')

        # Layer 3 Residuel block 1
        # x = self.identity_block(x, [64,64])
        # Layer 4 Residuel block 2
        x = self.convolutional_block(x, [64,64])
        # Layer 5 Residuel block 3
        # x = self.identity_block(x, [128,128])
        # Layer 6 Residuel block 4
        x = self.convolutional_block(x, [128,128])
        x = self.convolutional_block(x, [256,256])
        # Layer 7 Residuel block 5
        # x = self.identity_block(x, [128,128])
        
        # Layer 8 output block
        x = self.output_block(x,AvgPool=True,Flatten=True,GAP = False,FC=[512])

        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet_6")
        return model

    def CNN_Model(self):
        x_input = tf.keras.layers.Input(self.param.input_shape)
        x = self.input_block(x_input)

        x = self.cnn_block(x,channels=32, kernel_size=3, cn_strides=1, cn_padding='valid',
            BN=True,activation='relu',
            Pooling='Max',pool_size=2,pool_strides= 2,pool_padding = 'valid')
        x = self.cnn_block(x,channels=64, kernel_size=3, cn_strides=1, cn_padding='valid',
            BN=True,activation='relu',
            Pooling='Max',pool_size=2,pool_strides= 2,pool_padding = 'valid')
        x = self.cnn_block(x,channels=128, kernel_size=3, cn_strides=1, cn_padding='valid',
            BN=True,activation='relu',
            Pooling='Max',pool_size=2,pool_strides= 2,pool_padding = 'valid')

        x = self.output_block(x,AvgPool=False,Flatten=False,GAP = True,FC=None)
        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "CNN_3")
        return model

    def Build_Network(self):
        # self.model = self.CNN_Model()
        self.model = self.Residue_Model()

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.param.lr),
            loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
            metrics=['accuracy'])
        
        # Open the file
        if not os.path.exists(self.param.result):
            os.mkdir(self.param.result)

        with open(f'{self.param.result}summary_model_{self.param.version}.txt','w') as fh:
            # Pass the file handle in as a lambda function to make it callable
            self.model.summary(line_length=120,positions=[.40,.62,.75,1],print_fn=lambda x: fh.write(x + '\n')) 
    
    def Save_Model(self):
        model_json = self.model.to_json()

        plot_model(self.model, to_file=self.param.result+'model_'+str(self.param.version)+'.png', show_shapes=True)
        with open(self.param.result+'model_'+str(self.param.version)+'.json', 'w') as json_file:
            json_file.write(model_json)
            json_file.close()
        
    def Load_Model(self,loadweight=False):
        self.LoadIter()
        if loadweight:
            self.model = tf.keras.models.load_model(self.RUN_ITER+'Model')

    def Load_Data(self,mode='Train'):
        # Use Cache to increase efficiency
        # AUTOTUNE = tf.data.AUTOTUNE
        if mode=='Train':
            #train data
            train_ds =  tf.keras.utils.image_dataset_from_directory(
                    self.param.train_dir,
                    label_mode='categorical',
                    shuffle=True,
                    seed= self.param.seed,
                    image_size=(self.param.img_height, self.param.img_width),
                    batch_size=self.param.batch_size)
            # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
            return train_ds
        elif mode == 'Val':
            #val data
            val_ds = tf.keras.utils.image_dataset_from_directory(
                    self.param.val_dir,
                    label_mode='categorical',
                    shuffle=True,
                    seed= self.param.seed,
                    image_size=(self.param.img_height, self.param.img_width),
                    batch_size=self.param.batch_size)
            # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
            return val_ds
        elif mode == 'Test':
        #test data
            test_ds = tf.keras.utils.image_dataset_from_directory(
                    self.param.test_dir,
                    label_mode='categorical',
                    image_size=(self.param.img_height, self.param.img_width),
                    batch_size=1,
                    shuffle=False)
            # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
            return test_ds
    
    def LoadIter(self):
        if self.param.finetune:
            self.iter = self.param.finetune_iter
        else:
            if not os.path.exists(self.param.result+'iter.txt'):
                with open(self.param.result+'iter.txt', 'w') as result:
                    result.write("1")
                    result.close()
                self.iter = 1
            with open(self.param.result+'iter.txt', 'r') as result:
                self.iter = int(result.readline())
                result.close()
        self.RUN_ITER = self.param.result+'run_'+str(self.iter)+os.sep

    def Train(self):
        self.LoadIter()
        if not os.path.exists(self.RUN_ITER):
            os.mkdir(self.RUN_ITER)
        
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=self.RUN_ITER+'weights.hdf5',
            save_weights_only=True,
            monitor='val_accuracy',
            mode='max',
            save_best_only=True)
        # tb_callback = tf.keras.callbacks.TensorBoard(self.M_NAME+'logs', update_freq=1)
        csv_logger = tf.keras.callbacks.CSVLogger(self.RUN_ITER+f'history{"_finetune" if self.param.finetune else ""}.csv')
        progbar_logger = tf.keras.callbacks.ProgbarLogger(count_mode='steps')
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)

        train_ds = self.Load_Data(mode='Train')
        val_ds = self.Load_Data(mode='Val')
        curt = time.time()
        history = self.model.fit(
                x=train_ds,
                validation_data=val_ds,
                epochs=self.param.epoch,
                use_multiprocessing=True,
                workers=20,
                callbacks=[model_checkpoint_callback,csv_logger,progbar_logger,reduce_lr]
        )
        self.Train_Time = time.time()-curt
        print(f'Train Time: {self.Train_Time:.4f}')
        self.model.summary()

        self.Show_Results(history)
        self.Show_Results(history,'loss')
        
        out_name = glob.glob('*.out')[0]
        shutil.copy2(out_name,self.RUN_ITER+out_name.split(os.sep)[-1][:-4]+'.txt')
        open(out_name, 'w').close()

        err_name = glob.glob('*.err')[0]
        shutil.copy2(err_name,self.RUN_ITER+err_name.split(os.sep)[-1])
        open(err_name, 'w').close()

        with open(self.RUN_ITER+f'test_{self.iter}.txt', 'w') as result:
            result.write(f'train time: {self.Train_Time:.4f}\n')
            result.close()
        self.iter += 1
        self.SaveIter()
        self.model.save(self.RUN_ITER+'Model')

    def SaveIter(self):
        with open(self.param.result+'iter.txt', 'w') as result:
            result.write(str(self.iter))
            result.close()

    def Show_Results(self,history,acc_loss='accuracy'):
        plt.plot(history.history[acc_loss])
        plt.plot(history.history['val_'+acc_loss])
        ft = ' FineTune' if self.param.finetune else ''
        plt.title(f'Model v{self.param.version}{ft} {acc_loss}')
        plt.ylabel(acc_loss)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(self.RUN_ITER+f'{acc_loss}_r{self.iter}{ft}.jpg')
        plt.close()

    def Evaluate(self):
        test_ds = self.Load_Data(mode='Test')
        score = self.model.evaluate(x=test_ds,verbose=0)
        
        print('test loss:', score[0])
        print('test accuracy:', score[1])
        RUN_ITER = self.RUN_ITER+'test_'+str(self.iter-1)+'.txt'
        with open(RUN_ITER, 'w') as result:
            result.write(f'train time: {self.Train_Time:.4f}\ntest loss: {score[0]}\ntest accuracy: {score[1]}\n')
            result.close()

        