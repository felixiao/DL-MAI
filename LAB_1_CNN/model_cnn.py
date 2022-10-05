import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import argparse

import keras
from keras.utils import np_utils
#Define the NN architecture
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten
from keras.utils.vis_utils import plot_model

from sklearn.metrics import classification_report,confusion_matrix

import tensorflow as tf

class Config():
        name = 'MAMe'
        batch_size = 32
        img_height = 256
        img_width = 256
        channels = 3
        num_classes = 29
        path = '../datasets/'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs',dest='batchsize', type=int,help="set the batch_size; default is 32")
    parser.add_argument('-ep', dest='epochs',type=int,help="set the epochs; default is 10")
    # Parse arguments
    args = parser.parse_args()
    return args

class MAMe_CNN():
        def __init__(self,config,version=0):
                self.param = config
                self.Model_Ver = version
                self.PATH_METADATA = os.path.join(config.path,'MAMe_dataset.csv')
                self.PATH_LABEL = os.path.join(config.path,'MAMe_labels.csv')

                self.train_dir  = pathlib.Path(os.path.join(config.path,'train'))
                self.test_dir  = pathlib.Path(os.path.join(config.path,'test'))
                self.val_dir  = pathlib.Path(os.path.join(config.path,'val'))

                self.M_NAME = 'result'+os.sep+config.name+'_v'+str(self.Model_Ver)+os.sep
                if not os.path.exists(self.M_NAME):
                        os.mkdir(self.M_NAME)
                
                self.input_shape = (config.img_height, config.img_width,config.channels)

        def Load_Metadata(self):
                self.data = pd.read_csv(self.PATH_METADATA)
                self.classnames = list(pd.read_csv(self.PATH_LABEL,header=None,index_col=0)[1])
                self.labels = {self.classnames[i]:i for i in range(len(self.classnames))}
                print(self.labels)

        def Load_Data(self,seed=123):
                #train data
                self.train_ds = tf.keras.utils.image_dataset_from_directory(
                        self.train_dir,
                        label_mode='categorical',
                        class_names=self.classnames,
                        seed=seed,
                        image_size=(self.param.img_height, self.param.img_width),
                        batch_size=self.param.batch_size)

                #val data
                self.val_ds = tf.keras.utils.image_dataset_from_directory(
                        self.val_dir,
                        label_mode='categorical',
                        class_names=self.classnames,
                        seed= seed,
                        image_size=(self.param.img_height, self.param.img_width),
                        batch_size=self.param.batch_size)

                #test data
                self.test_ds = tf.keras.utils.image_dataset_from_directory(
                        self.test_dir,
                        label_mode='categorical',
                        image_size=(self.param.img_height, self.param.img_width),
                        batch_size=1,
                        class_names=self.classnames,
                        shuffle=False)
                
                # Use Cache to increase efficiency
                AUTOTUNE = tf.data.AUTOTUNE
                self.train_ds = self.train_ds.cache().prefetch(buffer_size=AUTOTUNE)
                self.val_ds = self.val_ds.cache().prefetch(buffer_size=AUTOTUNE)
                self.test_ds =self.test_ds.cache().prefetch(buffer_size=AUTOTUNE)

        def Build_Network(self):
                self.model = tf.keras.Sequential([
                        tf.keras.layers.Rescaling(1./255,input_shape=self.input_shape),
                        tf.keras.layers.Conv2D(32, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(32, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(32, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(128, activation='relu'),
                        tf.keras.layers.Dense(self.param.num_classes, activation=(tf.nn.softmax))
                ])

                self.model.compile(
                        optimizer='adam',
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

                self.model.summary()
                plot_model(self.model, to_file=self.M_NAME+'model.png', show_shapes=True)

        def Train(self,epochs=10):
                self.history = self.model.fit(
                        x=self.train_ds,
                        validation_data=self.val_ds,
                        epochs=epochs,
                        use_multiprocessing=True,
                        workers=6
                )
                #Saving model and weights

                model_json = self.model.to_json()
                with open(self.M_NAME+'model.json', 'w') as json_file:
                        json_file.write(model_json)
                weights_file = self.M_NAME+'weights.hdf5'
                self.model.save_weights(weights_file,overwrite=True)

                self.Show_Results()
                self.Show_Results('loss')

        def Show_Results(self,acc_loss='accuracy'):
                # 绘制训练 & 验证的准确率值/损失值
                plt.plot(self.history.history[acc_loss])
                plt.plot(self.history.history['val_'+acc_loss])
                plt.title('Model v'+str(self.Model_Ver)+' '+acc_loss)
                plt.ylabel(acc_loss)
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Val'], loc='upper left')
                plt.savefig(self.M_NAME+acc_loss+'.jpg')
                plt.close()

        def Evaluate(self):
                score = self.model.evaluate(x=self.test_ds,verbose=0)
                print('test loss:', score[0])
                print('test accuracy:', score[1])

if __name__ == '__main__':
        args = parse_arguments()
        config = Config()
        config.batch_size=args.batchsize

        print( 'Using Keras version', keras.__version__)
        mame_cnn = MAMe_CNN(config)
        print('Fin Init')
        mame_cnn.Load_Metadata()
        print('Fin Load Metadata')
        mame_cnn.Load_Data()
        print('Fin Load Data')
        mame_cnn.Build_Network()
        print('Fin Build_Network')
        mame_cnn.Train(args.epochs)
        print('Fin Train')
        mame_cnn.Evaluate()
        print('Fin Evaluate')