# modes:
# 1. create and train (and test)
# 2. create and save
# 3. load and train (and test)
# 4. load and test

import configparser
from model_mame import MAMe_CNN
import os
# import tensorflow as tf
import glob

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
        self.runs       = int(config['MAMe']['runs'])
        self.epoch      = int(config['MAMe']['epoch'])
        self.lr         = float(config['MAMe']['lr'])
        
        self.iter       = 1

        self.path       = config['PATH']['path']
        self.meta       = config['PATH']['meta']
        self.label      = config['PATH']['label']
        self.train_dir  = config['PATH']['train_dir']
        self.val_dir    = config['PATH']['val_dir']
        self.test_dir   = config['PATH']['test_dir']
        self.result     = config['PATH']['result']+str(self.version)+os.sep

        self.build  = config.getboolean('System', 'build')
        self.load   = config.getboolean('System', 'load')
        self.train  = config.getboolean('System', 'train')
        self.test   = config.getboolean('System', 'test')
        self.local  = config.getboolean('System', 'local')
        self.finetune  = config.getboolean('System', 'finetune')
        self.finetune_iter = int(config['System']['finetune_iter'])

if __name__ == '__main__':
    config= configparser.ConfigParser()
    config.read('config.ini')
    param = Param(config)

    if param.build and param.train:
        while param.iter <= param.runs:
            print(f'##################### Run {param.iter}/{param.runs}:')
            print('Build Network')
            mame_model = MAMe_CNN(param)
            mame_model.Build_Network()
            mame_model.Save_Model()
            mame_model.Train()
            param.iter += 1
            if param.test:
                mame_model.Evaluate()
        out_name = glob.glob('*.out')[0]
        os.remove(out_name)
        err_name = glob.glob('*.err')[0]
        os.remove(err_name)
    elif param.build:
        print('Build Network')
        mame_model = MAMe_CNN(param)
        mame_model.Build_Network()
        mame_model.Save_Model()
    elif param.load and param.train:
        while param.iter <= param.runs:
            print(f'##################### Run {param.iter}/{param.runs}:')
            print('Load Network')
            mame_model = MAMe_CNN(param)
            mame_model.Load_Model()
            mame_model.Save_Model()
            mame_model.Train()
            param.iter += 1
            if param.test:
                mame_model.Evaluate()
        out_name = glob.glob('*.out')[0]
        os.remove(out_name)
        err_name = glob.glob('*.err')[0]
        os.remove(err_name)
    elif param.finetune:
        print(f'##################### Run FineTune @ Iter = {param.finetune_iter}:')
        print('Init')
        mame_model = MAMe_CNN(param)
        print('Load Model and Weights')
        mame_model.Load_Model(loadweight=True)
        mame_model.Train()
        if param.test:
            mame_model.Evaluate()
        out_name = glob.glob('*.out')[0]
        os.remove(out_name)
        err_name = glob.glob('*.err')[0]
        os.remove(err_name)

