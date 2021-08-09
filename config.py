V0_CONFIG = {
    'name': 'V1',
    'n_relations': 50,
    'max_entity_pair_distance': 3
}

V1_CONFIG = {
    'name': 'V1',
    'n_relations': 50,
    'threshold': 0.8,
    'double_check': True,  # checks whether adding the relation improves the similarity or not
    'bilateral_context': 4,
    'max_entity_pair_distance': 3
}

V2_XGB_CONFIG = {
    'max_depth': 5,
    'colsample_bytree': 0.3,
    'n_estimators': 100,
    'learning_rate': 0.3
}

V2p5_XGB_CONFIG = {
    'max_depth': 5,
    'colsample_bytree': 0.3,
    'n_estimators': 100,
    'learning_rate': 0.3
}

V2_CONFIG = {
    'name': 'V2',
    'bilateral_context': 4,
    'max_entity_pair_distance': 3,
    'threshold': 0.9,
    'max_sentence_length': 12
}

V2p5_CONFIG = {
    'name': 'V2',
    'bilateral_context': 4,
    'max_entity_pair_distance': 3,
    'threshold': 0.6,
    'max_sentence_length': 16
}

V3_XGB_CONFIG = {
    'max_depth': 5,
    'colsample_bytree': 0.3,
    'n_estimators': 100,
    'learning_rate': 0.3
}

V3_CONFIG = {
    'name': 'V2',
    'bilateral_context': 4,
    'max_entity_pair_distance': 3,
    'threshold': 0.1,
    'max_sentence_length': 12
}

AMR_CONFIG = {
    'beam_size': 3,
    'entity_threshold': 0.8,
    'n_aliases': 3
}
