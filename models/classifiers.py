import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import f1_score
import os
from config import XGB_CONFIG


class XGBRelationClassifier(object):
    def __init__(self, experiment_name, config=XGB_CONFIG, load=False):
        """
        Creates a wrapper for the xgboost model\n
        :param experiment_name: name of the experiment: when using the save method, will save to
            experiment_results/experiment_name/
        :param config: configuration dictionary, by default will use the one in the config file
        :param load: if you want to load a previous experiment, load=True will attempt to load the config dictionary
            and the XGB model from the experiment name (the config parameter will be ignored in this case!)
        """
        self.experiment_name = experiment_name

        if load:
            with open('experiment_results/' + self.experiment_name + '/config.json', 'r') as f:
                self.config = json.load(f)
        else:
            self.config = config

        self.model = xgb.XGBClassifier(objective='multi:softmax',
                                       max_depth=config['max_depth'],
                                       colsample_bytree=config['colsample_bytree'],
                                       n_estimators=config['n_estimators'],
                                       learning_rate=config['learning_rate'],
                                       verbosity=0,
                                       use_label_encoder=False)
        if load:
            self.model.load_model('experiment_results/' + self.experiment_name + '/checkpoint.model')

        self.train_f1 = config['train_f1'] if 'train_f1' in config else -1
        self.val_f1 = config['val_f1'] if 'val_f1' in config else -1

    def train(self):
        """
        Trains the model on the training set and updates the micro-averaged F1 score attribute
        """
        Xt = np.load('wikidatavitals/data/encoded/train.npy')
        Yt = np.load('wikidatavitals/data/encoded/train_labels.npy').astype(int)
        self.model.fit(Xt, Yt)
        train_predictions = self.model.predict(Xt)
        self.train_f1 = f1_score(train_predictions, Yt, average='micro')
        self.config['train_f1'] = self.train_f1

    def val(self):
        """
        Updates the micro-averaged F1 score attribute on the validation set
        """
        Xv = np.load('wikidatavitals/data/encoded/val.npy')
        Yv = np.load('wikidatavitals/data/encoded/val_labels.npy').astype(int)
        val_predictions = self.model.predict(Xv)
        self.val_f1 = f1_score(val_predictions, Yv, average='micro')
        self.config['val_f1'] = self.val_f1

    def save(self):
        folder = 'experiment_results/' + self.experiment_name + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.model.save_model(folder + 'checkpoint.model')
        with open(folder + 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
