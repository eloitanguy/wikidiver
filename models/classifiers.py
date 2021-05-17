import numpy as np
import json
import xgboost as xgb
from sklearn.metrics import f1_score
import os


class XGBRelationClassifier(object):
    def __init__(self, config, experiment_name, load=''):
        self.config = config
        self.experiment_name = experiment_name

        self.model = xgb.XGBClassifier(objective='multi:softmax',
                                       max_depth=config['max_depth'],
                                       colsample_bytree=config['colsample_bytree'],
                                       n_estimators=config['n_estimators'],
                                       verbosity=0)

        if load != '':
            self.model.load_model(load)

    def train(self):
        """
        Trains the model and returns the micro-averaged F1 score on the training set
        """
        Xt = np.load(self.config['train_x_file'])
        Yt = np.load(self.config['train_y_file'])
        self.model.fit(Xt, Yt)
        train_predictions = np.argmax(self.model.predict(Xt), axis=1)
        return f1_score(train_predictions, Yt, average='micro')

    def val(self):
        """
        Returns the micro-averaged F1 score on the validation set
        """
        Xv = np.load(self.config['val_x_file'])
        Yv = np.load(self.config['val_y_file'])
        val_predictions = np.argmax(self.model.predict(Xv), axis=1)
        return f1_score(val_predictions, Yv, average='micro')

    def save(self):
        folder = 'experiment_results/' + self.experiment_name + '/'
        if not os.path.exists(folder):
            os.makedirs(folder)
        self.model.save_model(folder + 'checkpoint.model')
        with open(folder + 'config.json', 'w') as f:
            json.dump(self.config, f, indent=4)
