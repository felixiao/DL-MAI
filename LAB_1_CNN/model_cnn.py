import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import pathlib
import argparse
import time
import glob
import shutil

# import printfile as pf
import keras
from keras.utils import np_utils
#Define the NN architecture
from keras.models import Sequential,model_from_json
from keras.layers import Dense, Activation, Conv2D, MaxPooling2D, Flatten,Rescaling
from keras.losses import CategoricalCrossentropy
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
        seed = 123
        path = '../datasets/'

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-bs',dest='batchsize', type=int,help="set the batch_size; default is 32")
    parser.add_argument('-ep', dest='epochs',type=int,help="set the epochs; default is 10")
    parser.add_argument('-v', dest='version',type=int,help="set the version")
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

                if not os.path.exists(self.M_NAME+'iter.txt'):
                        self.runs = 1
                        with open(self.M_NAME+'iter.txt', 'w') as result:
                                result.write(str(self.runs))
                else:
                        with open(self.M_NAME+'iter.txt', 'r') as result:
                                self.runs = int(result.readline())
                                print('cur = ', self.runs)
                
                self.RUN_ITER = self.M_NAME+'run_'+str(self.runs)+'/'
                if not os.path.exists(self.RUN_ITER):
                        os.mkdir(self.RUN_ITER)

                # if not os.path.exists(self.RUN_ITER+'checkpoints/'):
                #         os.mkdir(self.RUN_ITER+'checkpoints/')
                
                self.input_shape = (config.img_height, config.img_width,config.channels)

        def Load_Metadata(self):
                self.metadata = pd.read_csv(self.PATH_METADATA)
                self.classnames = list(pd.read_csv(self.PATH_LABEL,header=None,index_col=0)[1])
                self.labels = {self.classnames[i]:i for i in range(len(self.classnames))}
                print(self.labels)

        def Load_Data(self,seed=123,mode='Train'):
                # Use Cache to increase efficiency
                # AUTOTUNE = tf.data.AUTOTUNE
                if mode=='Train':
                #train data
                        train_ds =  tf.keras.utils.image_dataset_from_directory(
                                self.train_dir,
                                label_mode='categorical',
                                shuffle=True,
                                seed= seed,
                                image_size=(self.param.img_height, self.param.img_width),
                                batch_size=self.param.batch_size)
                        # train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
                        return train_ds
                elif mode == 'Val':
                        #val data
                        val_ds = tf.keras.utils.image_dataset_from_directory(
                                self.val_dir,
                                label_mode='categorical',
                                shuffle=True,
                                seed= seed,
                                image_size=(self.param.img_height, self.param.img_width),
                                batch_size=self.param.batch_size)
                        # val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)
                        return val_ds
                elif mode == 'Test':
                #test data
                        test_ds = tf.keras.utils.image_dataset_from_directory(
                                self.test_dir,
                                label_mode='categorical',
                                image_size=(self.param.img_height, self.param.img_width),
                                batch_size=1,
                                shuffle=False)
                        # test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)
                        return test_ds


        def Build_Network(self):
                self.model = tf.keras.models.Sequential([
                        tf.keras.layers.Rescaling(1./255,input_shape=self.input_shape),
                        # RandomContrast(factor=0.2,seed=self.param.seed),
                        # RandomRotation(factor=0.2,fill_mode="reflect",seed=self.param.seed),
                        # RandomFlip(mode="horizontal_and_vertical",seed=self.param.seed),
                        tf.keras.layers.Conv2D(16, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(32, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Conv2D(64, 3, activation='relu'),
                        tf.keras.layers.MaxPooling2D(),
                        tf.keras.layers.Flatten(),
                        tf.keras.layers.Dense(512, activation='relu'),
                        tf.keras.layers.Dense(self.param.num_classes, activation=(tf.nn.softmax))
                ])

                self.model.compile(
                        optimizer=tf.keras.optimizers.SGD(learning_rate=0.01),
                        loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                        metrics=['accuracy'])

                self.model.summary()
                
                model_json = self.model.to_json()
                with open(self.M_NAME+'model.json', 'w') as json_file:
                        json_file.write(model_json)

                plot_model(self.model, to_file=self.M_NAME+'model.png', show_shapes=True)

        def Train(self,epochs=10):
                model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                        filepath=self.RUN_ITER+'weights.hdf5',
                        save_weights_only=True,
                        monitor='val_accuracy',
                        mode='max',
                        save_best_only=True)
                # tb_callback = tf.keras.callbacks.TensorBoard(self.M_NAME+'logs', update_freq=1)
                
                csv_logger = tf.keras.callbacks.CSVLogger(self.RUN_ITER+'history.csv')

                progbar_logger = tf.keras.callbacks.ProgbarLogger(count_mode='steps')

                # reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=0.001)

                train_ds = self.Load_Data(seed=123,mode='Train')

                val_ds = self.Load_Data(seed=123,mode='Val')
                curt = time.time()
                self.history = self.model.fit(
                        x=train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        use_multiprocessing=True,
                        workers=10,
                        callbacks=[model_checkpoint_callback,csv_logger,progbar_logger]
                )
                print('Train Time: ',time.time()-curt)
                
                self.Show_Results()
                self.Show_Results('loss')
                
                out_name = glob.glob('*.out')[0]
                shutil.move(out_name,self.RUN_ITER+out_name.split(os.sep)[-1][:-4]+'.txt')

                err_name = glob.glob('*.err')[0]
                shutil.move(err_name,self.RUN_ITER+err_name.split(os.sep)[-1])

                with open(self.M_NAME+'iter.txt', 'w') as result:
                        result.write(str(self.runs+1))
                

        def Show_Results(self,acc_loss='accuracy'):
                # 绘制训练 & 验证的准确率值/损失值
                plt.plot(self.history.history[acc_loss])
                plt.plot(self.history.history['val_'+acc_loss])
                plt.title('Model v'+str(self.Model_Ver)+' '+acc_loss)
                plt.ylabel(acc_loss)
                plt.xlabel('Epoch')
                plt.legend(['Train', 'Val'], loc='upper left')
                plt.savefig(self.RUN_ITER+acc_loss+'.jpg')
                plt.close()

        def Evaluate(self):
                test_ds = self.Load_Data(seed=123,mode='Test')
                score = self.model.evaluate(x=test_ds,verbose=0)
                print('test loss:', score[0])
                print('test accuracy:', score[1])

if __name__ == '__main__':
        args = parse_arguments()
        config = Config()
        config.batch_size=args.batchsize

        print( 'Using Keras version', keras.__version__)
        mame_cnn = MAMe_CNN(config,args.version)
        print('Fin Init')
        # mame_cnn.Load_Metadata()
        # print('Fin Load Metadata')
        # mame_cnn.Load_Data()
        # print('Fin Load Data')
        mame_cnn.Build_Network()
        print('Fin Build_Network')
        mame_cnn.Train(args.epochs)
        print('Fin Train')
        print('--------------------------END--------------------------')
        # mame_cnn.Evaluate()
        # print('Fin Evaluate')