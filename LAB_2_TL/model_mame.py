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

class MAMe_Base():
    def __init__(self, param):
        self.param = param
        os.environ['PYTHONHASHSEED']=str(param.seed)
        tf.random.set_seed(param.seed)
        np.random.seed(param.seed)
        random.seed(param.seed)
        self.strategy = tf.distribute.MirroredStrategy( param.GPUS )
        print('Number of devices: ', self.strategy.num_replicas_in_sync)

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
            self.model.load_weights(self.param.result+'weights.hdf5')

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

    def Confusion_Matrix_Model(self, mode):
        """ Evaluate given model and print results.
        Show validation loss and accuracy, classification report and 
        confusion matrix.

        Args:
            model (model): model to evaluate
            eval_gen (ImageDataGenerator): evaluation generator
        """
        # Confusion Matrix (validation subset)
        # eval_gen.reset()
        ds = self.Load_Data(mode)
        pred = self.model.predict(ds, verbose=0,batch_size=32,workers=20,use_multiprocessing=True)

        # Assign most probable label
        predicted_class_indices = np.argmax(pred,axis=1)

        # Get class labels
        # labels = (ds.class_indices)
        target_names = labels.keys()

        # Plot statistics
        print(classification_report(self.param.num_classes, predicted_class_indices, target_names=target_names))

        cf_matrix = confusion_matrix(np.array(self.param.num_classes), predicted_class_indices)
        fig, ax = plt.subplots(figsize=(13, 13)) 
        sns.heatmap(cf_matrix, annot=True, cmap='PuRd', cbar=False, square=True, xticklabels=target_names, yticklabels=target_names)
        # plt.show()
        fname = self.RUN_ITER+mode+'_'+str(self.iter-1)+'.jpg'
        plt.savefig(fname)

    def Plot_Result(self,hist,fn='FE',FT_Epoch=0):
        plt.figure(figsize=(8, 8))
        plt.subplot(2, 1, 1)
        dim = np.arange(1,len(hist.history['accuracy'])+1,1)
        plt.plot(dim,hist.history['accuracy'], label='Training Accuracy')
        plt.plot(dim,hist.history['val_accuracy'], label='Validation Accuracy')
        if FT_Epoch>1:
            plt.plot([FT_Epoch,FT_Epoch],plt.ylim([0,1.0]), label='Start Fine Tuning')
        plt.legend(loc='lower right')
        plt.ylabel('Accuracy')
        plt.ylim([min(plt.ylim()),1])
        plt.title('Training and Validation Accuracy')

        plt.subplot(2, 1, 2)
        plt.plot(dim,hist.history['loss'], label='Training Loss')
        plt.plot(dim,hist.history['val_loss'], label='Validation Loss')
        if FT_Epoch>1:
            plt.plot([FT_Epoch,FT_Epoch],plt.ylim(), label='Start Fine Tuning')
        plt.legend(loc='upper right')
        plt.ylabel('Cross Entropy')
        # plt.ylim([0,1.0])
        plt.title('Training and Validation Loss')
        plt.xlabel('epoch')
        plt.tight_layout()
        plt.savefig(self.param.result+f'result_{fn}.jpg')
        plt.close()

    def GetLayerInfo(self,txt,namelist):
      n = txt.strip().split(' ')[0]
      if n == 'Layer':
        return 'Trainable'
      if n in namelist:
        return 'Trainable' if self.model.get_layer(n).trainable else 'Freeze'
      return ''

    def Model_Summary(self,fn='summary_model_FE',model=None):
      # Open the file
      if not os.path.exists(self.param.result):
          os.mkdir(self.param.result)

      with open(f'{self.param.result}{fn}.txt','w') as fh:
          # Pass the file handle in as a lambda function to make it callable
          if model:
            model.summary(line_length=120,positions=[.40,.62,.75,1],print_fn=lambda x: fh.write(x + self.GetLayerInfo(x,self.baselayernames)+'\n'),show_trainable=True)
          else:
            self.model.summary(line_length=120,positions=[.40,.62,.75,1],print_fn=lambda x: fh.write(x + self.GetLayerInfo(x,self.layernames)+'\n'))
      
    #   self.model.summary()

    def CallBacks(self):
      self.early_stop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', 
            min_delta=0.00001, 
            patience=10, 
            # verbose=1, 
            mode='auto', 
            restore_best_weights=True)

      self.model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
          filepath=self.param.result+'weights.hdf5',
          save_weights_only=False,
          monitor='val_accuracy',
          mode='auto',
          # verbose=1, 
          save_best_only=True)
      # tb_callback = tf.keras.callbacks.TensorBoard(self.RUN_ITER+'logs', update_freq=1)
      self.csv_logger = tf.keras.callbacks.CSVLogger(self.param.result+f'history.csv')
      self.progbar_logger = tf.keras.callbacks.ProgbarLogger(count_mode='steps')
      self.reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-7)

    def TrainModel(self,epoch,init_epoch=0):
        curt = time.time()
        
        hist = self.model.fit(
                x=self.train_ds,
                validation_data=self.val_ds,
                epochs=epoch,
                initial_epoch=init_epoch,
                use_multiprocessing=True,
                workers=20,
                callbacks=[self.model_checkpoint_callback,self.csv_logger,self.progbar_logger,self.reduce_lr,self.early_stop_callback]
        )

        traintime = time.time()-curt
        print(f'FE Train Time: {traintime:.4f}')

        return hist, traintime
        # self.model.save(self.param.result+'Model')
    
    def CreateModel(self):
        raise NotImplementedError

    def FeatureExtraction(self):
        raise NotImplementedError
    
    def FineTune(self):
        raise NotImplementedError

class MAMe_InceptionResNetV2(MAMe_Base):
    def __init__(self, param):
        super(MAMe_Base, self).__init__(param)
    
    def CreateAndFineTune(self):
        print(f'##################### Create PreTrain {self.param.pretrain_name} V{self.param.version}#####################')
        with self.strategy.scope():
            self.base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
                include_top=False,
                weights=os.path.join(self.param.pretrain_dir,self.param.pretrain_fn),
                input_shape=self.param.input_shape,
                pooling=None)
            # self.base_model = tf.keras.applications.vgg16.VGG16(
            #     include_top=False,
            #     weights=model_path,
            #     input_shape=self.param.input_shape,
            #     pooling=None)

            self.base_model.trainable = True

            x_input = tf.keras.layers.Input(self.param.input_shape)
            x = tf.keras.layers.Resizing(height=224,width=224)(x_input)
            x = tf.keras.layers.RandomFlip(seed=self.param.seed)(x)
            x = tf.keras.layers.RandomRotation(seed=self.param.seed,factor=0.2, fill_mode='reflect')(x)

            x = tf.keras.applications.resnet_v2.preprocess_input(x)
            # for layer in self.base_model.layers[:-100]:
            #   layer.trainable = False

            x = self.base_model(x)
            
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            # x = tf.keras.layers.Flatten()(x)
            # x = tf.keras.layers.Dense(256, activation = tf.nn.relu,name='FC')(x)
            x = tf.keras.layers.Dropout(self.param.FE_out_dropout)(x)
            predictions = tf.keras.layers.Dense(self.param.num_classes, activation = tf.nn.softmax,name='Predictions')(x)
            self.model = tf.keras.Model(x_input, outputs=predictions,name=self.param.name)
            self.Model_Summary(fn='base_model',model=self.base_model)

            optim = tf.keras.optimizers.Adam(learning_rate=self.param.FT_lr)
            if self.param.FT_optimizer == 'SGD':
                optim = tf.keras.optimizers.SGD(learning_rate=self.param.FT_lr)
            elif self.param.FT_optimizer == 'RMSprop':
                optim = tf.keras.optimizers.RMSprop(learning_rate=self.param.FT_lr)
            self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = optim,
              metrics=['accuracy'])
        
        print(f'Number of final layers: {len(self.model.layers)}')
        print(f'Number of trainable variables: {len(self.model.trainable_variables)}')
        self.Model_Summary(fn='summary_model_FT')
        self.csv_logger = tf.keras.callbacks.CSVLogger(self.param.result+f'history.csv')
        
        self.train_ds = self.Load_Data(mode='Train')
        self.val_ds = self.Load_Data(mode='Val')
        self.CallBacks()

        self.history, self.Train_Time= self.TrainModel(self.param.FT_epoch)
        
        self.model.save(self.param.result+'Model')

        self.Plot_Result(self.history,fn='FineTune')
        self.Evaluate()

    def CreateModel(self):
        print(f'##################### Create PreTrain Model {self.param.pretrain_name} V{self.param.version} #####################')
        with self.strategy.scope():
            self.base_model = tf.keras.applications.inception_resnet_v2.InceptionResNetV2(
                include_top=False,
                weights=os.path.join(self.param.pretrain_dir,self.param.pretrain_fn),
                input_shape=self.param.input_shape,
                pooling=None)

            print(self.param.pretrain_name,'Layers:',len(self.base_model.layers))
            # Feature extraction
            self.base_model.trainable = False
            # self.base_model.summary()

            x_input = tf.keras.layers.Input(self.param.input_shape,name='Input')
            x = tf.keras.layers.Resizing(height=224,width=224)(x)
            x = tf.keras.layers.RandomFlip(seed=self.param.seed)(x)
            x = tf.keras.layers.RandomRotation(seed=self.param.seed,factor=0.2, fill_mode='reflect')(x)
            x = tf.keras.applications.resnet_v2.preprocess_input(x)
            x = self.base_model(x,training=False)
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            # # x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256, activation = tf.nn.relu,name='FC')(x)
            x = tf.keras.layers.Dropout(self.param.FE_out_dropout)(x)
            predictions = tf.keras.layers.Dense(self.param.num_classes, activation = tf.nn.softmax,name='Predictions')(x)

            self.model = tf.keras.Model(x_input, outputs=predictions,name=self.param.pretrain_name)
            
            # self.model = tf.keras.Sequential()
            # self.model.add(tf.keras.layers.Input(self.param.input_shape,name='Input'))
            # self.model.add(tf.keras.layers.Resizing(height=self.param.img_height,width=self.param.img_width,name='Resize'))
            # self.model.add(tf.keras.layers.RandomFlip(seed=self.param.seed,name='RandFlip'))
            # self.model.add(tf.keras.layers.RandomRotation(seed=self.param.seed,factor=0.2, fill_mode='reflect',name='RandRot'))
            # self.model.add(tf.keras.layers.Rescaling(1./255,name='Rescaling'))

            # for layer in self.base_model.layers[1:]:
            #   layer.trainable=False
            #   self.model.add(layer)

            # self.model.add(tf.keras.layers.GlobalAveragePooling2D(name='GAvgPool2D'))
            # self.model.add(tf.keras.layers.Flatten(name='Flatten'))
            # self.model.add(tf.keras.layers.Dense(256, activation = tf.nn.relu,name='FC'))
            # self.model.add(tf.keras.layers.Dropout(self.param.FE_out_dropout,name='Dropout'))
            # self.model.add(tf.keras.layers.Dense(self.param.num_classes, activation = tf.nn.softmax,name='Predictions'))
        
        self.layernames = [l.name for l in self.model.layers]
        self.baselayernames = [l.name for l in self.base_model.layers]
        self.train_ds = self.Load_Data(mode='Train')
        self.val_ds = self.Load_Data(mode='Val')

    def FeatureExtraction(self):
        print('##################### Feature Extraction Param #####################')
        print(f'''
            BatchSize: {self.param.batch_size}
            Optimizer: {self.param.FE_optimizer}
            LR:        {self.param.FE_lr}
            Epoch:     {self.param.FE_epoch}''')
        print('##################### Feature Extraction #####################')
        with self.strategy.scope():
            optim = tf.keras.optimizers.Adam(learning_rate=self.param.FE_lr)
            if self.param.FE_optimizer == 'SGD':
                optim = tf.keras.optimizers.SGD(learning_rate=self.param.FE_lr)
            elif self.param.FE_optimizer == 'RMSprop':
                optim = tf.keras.optimizers.RMSprop(learning_rate=self.param.FE_lr)
            
            self.model.compile(
                optimizer=optim,
                loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

        # Print model summary
        print(f'Number of final layers: {len(self.model.layers)}')
        print(f'Number of trainable variables: {len(self.model.trainable_variables)}')

        self.Model_Summary()
        
        self.CallBacks()
    
        curt = time.time()
        self.csv_logger = tf.keras.callbacks.CSVLogger(self.param.result+f'history.csv')
        self.history, traintime = self.TrainModel(self.param.FE_epoch)
        self.initial_epoch = self.history.epoch[-1]+1
        
        self.Plot_Result(self.history_FE,fn='FeatureExtract')

    def FineTune(self):
        print('##################### FineTune Param #####################')
        print(f'''
            BatchSize: {self.param.batch_size}
            Optimizer: {self.param.FT_optimizer}
            LR:        {self.param.FT_lr}
            Epoch:     {self.param.FT_epoch}
            Trainable Layers: {self.param.FT_train_layer}/{len(self.base_model.layers)}''')
        print('##################### FineTune #####################')
        with self.strategy.scope():
            x_input = tf.keras.layers.Input(self.param.input_shape)
            x = tf.keras.layers.Resizing(height=224,width=224)(x)
            x = tf.keras.layers.RandomFlip(seed=self.param.seed)(x)
            x = tf.keras.layers.RandomRotation(seed=self.param.seed,factor=0.2, fill_mode='reflect')(x)

            x = tf.keras.applications.resnet_v2.preprocess_input(x)
            x = self.base_model(x,training=False)
            train_layers = ['conv_7b_bn','conv_7b','block8_10_conv','batch_normalization_405','batch_normalization_402','conv2d_402','conv2d_405',
                            'batch_normalization_404','conv2d_404','batch_normalization_403','conv2d_403']
            for layer in self.base_model.layers:
              if layer.name ==train_layers:
                    layer.trainable = True
              else:
                    layer.trainable = False
            x = tf.keras.layers.GlobalAveragePooling2D()(x)
            # x = tf.keras.layers.Flatten()(x)
            x = tf.keras.layers.Dense(256, activation = tf.nn.relu,name='FC')(x)
            x = tf.keras.layers.Dropout(self.param.FE_out_dropout)(x)
            predictions = tf.keras.layers.Dense(self.param.num_classes, activation = tf.nn.softmax,name='Predictions')(x)
            self.model = tf.keras.Model(x_input, outputs=predictions,name=self.param.name)
            self.Model_Summary(fn='base_model',model=self.base_model)

            # last_layer=False
            # for layer in self.model.layers:
            #     if layer.name =='block5_conv2' or last_layer:
            #         layer.trainable = True
            #         last_layer = True
            #     else:
            #         layer.trainable = False

            optim = tf.keras.optimizers.Adam(learning_rate=self.param.FT_lr)
            if self.param.FT_optimizer == 'SGD':
                optim = tf.keras.optimizers.SGD(learning_rate=self.param.FT_lr)
            elif self.param.FT_optimizer == 'RMSprop':
                optim = tf.keras.optimizers.RMSprop(learning_rate=self.param.FT_lr)
            self.model.compile(loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
              optimizer = optim,
              metrics=['accuracy'])

        self.Model_Summary(fn='summary_model_FE')
        print(f'Number of trainable variables: {len(self.model.trainable_variables)}')

        self.csv_logger = tf.keras.callbacks.CSVLogger(self.param.result+f'history.csv',append=True)
        self.history, traintime= self.TrainModel(self.param.FT_epoch,self.initial_epoch)
        
        self.model.save(self.param.result+'Model')

        self.Plot_Result(self.history_FE,fn='FineTune',FT_Epoch=self.initial_epoch)