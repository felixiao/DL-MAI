import pandas as pd
import numpy as np
import os
os.environ['PYTHONHASHSEED']=str(123)

import matplotlib.pyplot as plt
import time
import glob
import shutil
import random
import configparser
import tensorflow as tf

#Define the NN architecture
from keras.utils.vis_utils import plot_model
import seaborn as sns
from sklearn.metrics import classification_report,confusion_matrix
from model_mame import MAMe_CNN,MAMe_InceptionResNetV2


class Param():
    def __init__(self, config):
        self.name = config['MAMe']['name']
        self.batch_size = int(config['MAMe']['batch_size'])
        self.img_height = int(config['MAMe']['img_height'])
        self.img_width  = int(config['MAMe']['img_width'])
        self.channels   = int(config['MAMe']['channels'])
        self.input_shape= (self.img_height, self.img_width, self.channels)
        self.num_classes= int(config['MAMe']['num_classes'])
        self.seed       = int(config['MAMe']['seed'])
        self.version    = int(config['MAMe']['version'])

        self.path       = config['PATH']['path']
        self.meta       = config['PATH']['meta']
        self.label      = config['PATH']['label']
        self.train_dir  = config['PATH']['train_dir']
        self.val_dir    = config['PATH']['val_dir']
        self.test_dir   = config['PATH']['test_dir']
        self.result     = config['PATH']['result']+str(self.version)+os.sep
        self.pretrain_dir=config['PATH']['pretrain_dir']
        self.pretrain_name=config['PATH']['pretrain_name']
        self.pretrain_fn=config['PATH']['pretrain_fn']

        self.FE_epoch   = int(config['FeatureExtract']['epoch'])
        self.FE_optimizer=config['FeatureExtract']['optimizer']
        self.FE_lr      = float(config['FeatureExtract']['lr'])
        self.FE_out_pooling= config['FeatureExtract']['out_pooling']
        self.FE_out_dropout= float(config['FeatureExtract']['out_dropout'])

        self.FT_epoch   = int(config['FineTune']['epoch'])
        self.FT_optimizer=config['FineTune']['optimizer']
        self.FT_lr      = float(config['FineTune']['lr'])
        self.FT_train_layer= int(config['FineTune']['train_layer'])


        self.build  = config.getboolean('System', 'build')
        self.train  = config.getboolean('System', 'train')
        self.test   = config.getboolean('System', 'test')
        self.gpu_count = int(config['System']['gpu_count'])
        self.GPUS = [f'GPU:{i}' for i in range(self.gpu_count)]

def set_reproducible(seed=2022):
    os.environ['PYTHONHASHSEED']=str(param.seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed) #analogue of set_random_seed(seed_value)
    # tf.random.uniform([1], seed=seed)
    # torch.manual_seed(seed)
    # torch.use_deterministic_algorithms(True)

if __name__ == '__main__':
    config= configparser.ConfigParser()
    config.read('config.ini')
    param = Param(config)
    set_reproducible(param.seed)

    mame_model = MAMe_InceptionResNetV2(param)
    mame_model.CreateAndFineTune()