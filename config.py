V1_CONFIG = {
    'n_relations': 50,
    'threshold': 0.8,
    'double_check': True,
    'bilateral_context': 4
}

XGB_config = {
    'train_x_file': 'wikidatavitals/data/encoded/train.npy',
    'train_y_file': 'wikidatavitals/data/encoded/train_labels.npy',
    'val_x_file': 'wikidatavitals/data/encoded/val.npy',
    'val_y_file': 'wikidatavitals/data/encoded/val_labels.npy',
    'max_depth': 10,
    'colsample_bytree': 0.7,
    'n_estimators': 100
}
