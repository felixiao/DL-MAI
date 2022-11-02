import pandas as pd
import numpy as np
import os
os.environ['PYTHONHASHSEED']=str(123)

import matplotlib.pyplot as plt
import time
import glob
import shutil
import random
import tensorflow as tf
#Define the NN architecture
from keras.utils.vis_utils import plot_model

class MAMe_CNN():
    def __init__(self, param):
        self.param = param
        os.environ['PYTHONHASHSEED']=str(123)
        tf.random.set_seed(param.seed)
        np.random.seed(param.seed)
        random.seed(param.seed)

    def cnn_block(self,x, channels=32,BN=True,axis=-1,activation='relu', Pooling='Max',kernel_size=3, cn_strides=1,cn_padding = 'valid',pool_size=2,pool_strides=2,pool_padding='valid'):
        x = tf.keras.layers.Conv2D(channels, kernel_size=kernel_size, strides=cn_strides, padding=cn_padding)(x)
        if BN:
            x = tf.keras.layers.BatchNormalization(axis=axis)(x)
        if activation:
            x = tf.keras.layers.Activation(activation)(x)
        if Pooling == 'Max':
            x = tf.keras.layers.MaxPool2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)(x)
        elif Pooling == 'Avg':
            x = tf.keras.layers.AveragePooling2D(pool_size=pool_size, strides=pool_strides, padding=pool_padding)(x)
        return x
    
    def input_block(self,x,agu_flip=True,agu_rot=True):
        x = tf.keras.layers.Rescaling(1./255)(x)
        x = tf.keras.layers.ZeroPadding2D((3, 3))(x)
        if agu_flip:
            x = tf.keras.layers.RandomFlip(seed=self.param.seed)(x)
        if agu_rot:
            x = tf.keras.layers.RandomRotation(seed=self.param.seed,factor=0.2, fill_mode='reflect')(x)
        return x

    def output_block(self,x,Pooling='Avg',Flatten=False,GAP=True,FC=None,activation=tf.nn.softmax):
        if Pooling=='Avg':
            x = tf.keras.layers.AveragePooling2D((2,2), padding = 'same')(x)
        elif Pooling == 'Max':
            x = tf.keras.layers.MaxPooling2D((2,2), padding = 'same')(x)
        if Flatten:
            x = tf.keras.layers.Flatten()(x)
        if GAP:
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
        if FC:
            for s in FC:
                x = tf.keras.layers.Dense(s, activation = 'relu')(x)
        x = tf.keras.layers.Dense(self.param.num_classes, activation = activation)(x)
        return x


    def res_block(self,x, filters=[32,32],kernel_size=3, downsample=None):
        """_summary_

        Args:
            x (_type_): the inputs
            filters (list, optional): list of number of filters for each conv layers. Defaults to [32,32].
            kernel_size (int, optional): kernel size for all conv layers. Defaults to 3.
            downsample (_type_, optional): use downsample or not. Defaults to None, also 'Conv', 'MaxPool' .

        Returns:
            _type_: the outputs
        """
        # copy tensor to variable called x_skip
        x_skip = x
        for i,f in enumerate(filters):
            # no activation for the last conv block
            x = self.cnn_block(x,channels=f,kernel_size=kernel_size,cn_padding='same',cn_strides=2 if downsample=='Conv' and i ==0 else 1,
                BN=True,activation='relu' if i != len(filters)-1 else None,
                Pooling='Max' if i ==0 and downsample=='MaxPool' else None, pool_size=2,pool_strides= 2)

        if downsample == 'MaxPool':
            # Downsampling with MaxPool2D
            x_skip = tf.keras.layers.MaxPool2D(pool_size=2,strides=2)(x_skip)
        elif downsample == 'Conv':
            # Processing Residue with conv(1,1)
            x_skip = tf.keras.layers.Conv2D(filters[-1], 1, strides = 2)(x_skip)

        # Add Residue
        x = tf.keras.layers.Add()([x, x_skip])
        x = tf.keras.layers.Activation('relu')(x)
        return x

    
    def Residue_Model(self):
        x_input = tf.keras.layers.Input(self.param.input_shape)
        # Layer 1 Input and Agumentaion Layers
        x = self.input_block(x_input)

        # Layer 2 (Initial Conv layer along with maxPool)
        x = self.cnn_block(x,channels=64, kernel_size=7, cn_strides=2, cn_padding='same',
            BN=True,axis=-1,activation='relu',
            Pooling='Max',pool_size=3,pool_strides= 2,pool_padding = 'same')

        # Residuel blocks 
        x = self.res_block(x, [64,64])
        x = self.res_block(x, [64,64])
        x = self.res_block(x, [64,64])

        x = self.res_block(x, [128,128],downsample='Conv')
        x = self.res_block(x, [128,128])
        x = self.res_block(x, [128,128])
        x = self.res_block(x, [128,128])

        x = self.res_block(x, [256,256],downsample='Conv')
        x = self.res_block(x, [256,256])
        x = self.res_block(x, [256,256])
        x = self.res_block(x, [256,256])
        x = self.res_block(x, [256,256])
        x = self.res_block(x, [256,256])

        x = self.res_block(x, [512,512],downsample='Conv')
        x = self.res_block(x, [512,512])
        x = self.res_block(x, [512,512])
        
        # Output block
        x = self.output_block(x,Pooling=self.param.out_Pool,Flatten=self.param.out_Flatten,GAP = self.param.out_GAP,FC=None if self.param.out_FC==0 else [self.param.out_FC])

        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "ResNet_34")
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

        x = self.output_block(x,Pooling=self.param.out_Pool,Flatten=self.param.out_Flatten,GAP = self.param.out_GAP,FC=None if self.param.out_FC==0 else [self.param.out_FC])
        model = tf.keras.models.Model(inputs = x_input, outputs = x, name = "CNN_3")
        return model

    def Build_Network(self):
        # self.model = self.CNN_Model()
        self.model = self.Residue_Model()
        optimizer = tf.keras.optimizers.Adam(learning_rate=self.param.lr)
        if self.param.optimizer == 'SGD':
            optimizer = tf.keras.optimizers.SGD(learning_rate=self.param.lr)
        self.model.compile(
            optimizer=optimizer,
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
            self.model = tf.keras.models.model_from_json(open(f'{self.param.result}model_{self.param.version}.json').read())
            self.model.load_weights(self.RUN_ITER+'weights.hdf5')

            optimizer = tf.keras.optimizers.Adam(learning_rate=self.param.lr)
            if self.param.optimizer == 'SGD':
                optimizer = tf.keras.optimizers.SGD(learning_rate=self.param.lr)
            self.model.compile(
                optimizer=optimizer,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])
            # self.model = tf.keras.models.load_model(self.RUN_ITER+'Weights.hdf5')

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
        # tb_callback = tf.keras.callbacks.TensorBoard(self.RUN_ITER+'logs', update_freq=1)
        csv_logger = tf.keras.callbacks.CSVLogger(self.RUN_ITER+f'history{"_finetune" if self.param.finetune else ""}.csv')
        progbar_logger = tf.keras.callbacks.ProgbarLogger(count_mode='steps')
        reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)

        train_ds = self.Load_Data(mode='Train')
        val_ds = self.Load_Data(mode='Val')
        curt = time.time()
        self.history = self.model.fit(
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

        self.Show_Results()
        self.Show_Results('loss')
        
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

    def Show_Results(self,acc_loss='accuracy'):
        plt.plot(self.history.history[acc_loss])
        plt.plot(self.history.history['val_'+acc_loss])
        ft = ' FineTune' if self.param.finetune else ''
        plt.title(f'Model v{self.param.version}{ft} {acc_loss}')
        plt.ylabel(acc_loss)
        plt.xlabel('Epoch')
        plt.legend(['Train', 'Val'], loc='upper left')
        plt.savefig(self.RUN_ITER+f'{acc_loss}_r{self.iter}{ft}.jpg')
        plt.close()

    def Evaluate(self,use_best=True):
        if use_best:
            self.model.load_weights(self.RUN_ITER+'weights.hdf5')
        csv_logger = tf.keras.callbacks.CSVLogger(self.RUN_ITER+f'test{"_finetune" if self.param.finetune else ""}.csv')
        test_ds = self.Load_Data(mode='Test')
        score = self.model.evaluate(x=test_ds,verbose=1,workers=20,use_multiprocessing=True,callbacks=[csv_logger])
        
        print('test loss:', score[0])
        print('test accuracy:', score[1])
        RUN_ITER = self.RUN_ITER+'test_'+str(self.iter-1)+'.txt'
        with open(RUN_ITER, 'w') as result:
            result.write(f'train time: {self.Train_Time:.4f}\ntest loss: {score[0]}\ntest accuracy: {score[1]}\n')
            result.close()
