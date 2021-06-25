from models.classifiers import XGBRelationClassifier
from models.comparators import get_sliced_relation_mention
import argparse
from config import V3_CONFIG
import json
import torch
from models.encoders import PairEncoder
import numpy as np
from benchmark import usa_benchmark, hundo_benchmark
from v2 import train_v2
from extractor_base import Extractor, NoFact


class V3(Extractor):
    """
    Upon creation, loads a trained XGB model (trained using v3.py using the parameters in the config file).\n
    This model classifies entity pairs (ie a pair of word groups detected by wikifier) into relations. The word pairs
    are encoded using BERT's attentions, and we classify these pair encodings using an XGB model.
    The training procedure uses the configuration in V3_XGB_CONFIG, and this model uses the parameters in V3_CONFIG.
    """
    def __init__(self, experiment_name='v3_trained'):
        super().__init__(max_entity_pair_distance=V3_CONFIG['max_entity_pair_distance'])
        self.xgb = XGBRelationClassifier(experiment_name, load=True, model_type='v3')
        self.bilateral_context = V3_CONFIG['bilateral_context']
        self.threshold = V3_CONFIG['threshold']
        self.max_sentence_length = V3_CONFIG['max_sentence_length']

        with open('wikidatavitals/data/encoded/relation_indices.json', 'r') as f:
            self.relations_by_idx = json.load(f)

        self.device = torch.device('cuda:0')
        self.PE = PairEncoder(max_sentence_length=self.max_sentence_length)

    def _get_relation(self, e1_dict, e2_dict, processed_text):
        sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, processed_text,
                                                      bilateral_context=self.bilateral_context)

        if len(sliced_sentence.split(' ')) > self.max_sentence_length:
            raise NoFact

        e1_slice = e1_dict['start_idx'], e1_dict['end_idx'] + 1
        e2_slice = e2_dict['start_idx'], e2_dict['end_idx'] + 1
        xgb_input = self.PE.get_pair_encoding(sliced_sentence.split(' '), e1_slice, e2_slice).cpu().numpy()
        xgb_input = xgb_input.reshape((1, 288))  # unclear why this is necessary, original shape is (288,)

        # Computing XGB output
        probabilities = self.xgb.model.predict_proba(xgb_input)[0]  # shape (50)
        chosen_idx = np.argmax(probabilities)
        best_probability = probabilities[chosen_idx]

        if best_probability > self.threshold:
            return self.relations_by_idx[chosen_idx]['id']

        raise NoFact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', '--train', action='store_true', help='Trains and saves an XGB classifier ')
    parser.add_argument('--s', '--sentence', type=str, default='', help="Sentence to extract facts from using v3")
    parser.add_argument('--u-b', '--usa-benchmark', action='store_true', help="Run the USA benchmark")
    parser.add_argument('--h-b', '--hundo-benchmark', action='store_true', help="Run the Hundo benchmark")
    args = parser.parse_args()

    if args.s:
        v3 = V3()
        print('Parsing "{}" ...'.format(args.s))
        v3.extract_facts(args.s, verbose=True)

    if args.t:
        print('Training v3 ...')
        train_v2(model_type='v3')

    if args.u_b:
        v3 = V3()
        usa_benchmark(v3, V3_CONFIG)

    if args.h_b:
        v3 = V3()
        hundo_benchmark(v3, V3_CONFIG)
