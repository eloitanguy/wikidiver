import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import f1_score
import os
from config import V2_XGB_CONFIG, V2p5_XGB_CONFIG, V3_XGB_CONFIG


class XGBRelationClassifier(object):
    def __init__(self, experiment_name, config=None, load=False, model_type='v2'):
        """
        Creates a wrapper for the xgboost model\n
        :param experiment_name: name of the experiment: when using the save method, will save to
            experiment_results/experiment_name/
        :param config: configuration dictionary, by default (None) will use the one in the config file
        :param load: if you want to load a previous experiment, load=True will attempt to load the config dictionary
            and the XGB model from the experiment name (the config parameter will be ignored in this case!)
        """
        self.experiment_name = experiment_name
        assert model_type in ['v2', 'v2.5', 'v3'], 'model_type: {} is not in [v2, v2.5, v3]'.format(model_type)
        model_type_to_config = {'v2': V2_XGB_CONFIG, 'v2.5': V2p5_XGB_CONFIG, 'v3': V3_XGB_CONFIG}
        if config is None:
            config = model_type_to_config[model_type]
        self.model_type = model_type

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

        if self.model_type == 'v2':  # wikidatavitals pseudo-sentences
            self.dataset_folder = 'wikidatavitals/data/encoded/'
        else:  # wikivitals annotated sentences
            self.dataset_folder = 'wikivitals/data/encoded/'

    def train(self, verbose=False):
        """
        Trains the model on the training set and updates the micro-averaged F1 score attribute
        """
        Xt = np.load(self.dataset_folder + 'train.npy')
        Yt = np.load(self.dataset_folder + 'train_labels.npy').astype(int)
        self.model.fit(Xt, Yt)
        train_predictions = self.model.predict(Xt)
        self.train_f1 = f1_score(train_predictions, Yt, average='micro')
        self.config['train_f1'] = self.train_f1
        if verbose:
            print('Finished training! Train F1: {:.2f}%'.format(self.train_f1 * 100))

    def val(self, verbose=False):
        """
        Updates the micro-averaged F1 score attribute on the validation set
        """
        Xv = np.load(self.dataset_folder + 'val.npy')
        Yv = np.load(self.dataset_folder + 'val_labels.npy').astype(int)
        val_predictions = self.model.predict(Xv)
        self.val_f1 = f1_score(val_predictions, Yv, average='micro')
        self.config['val_f1'] = self.val_f1
        if verbose:
            print('Finished validation! Val F1: {:.2f}%'.format(self.val_f1 * 100))

    def save(self):
        folder = 'experiment_results/' + self.experiment_name + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.model.save_model(folder + 'checkpoint.model')
        self.config['model_type'] = self.model_type
        with open(folder + 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
