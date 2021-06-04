from models.classifiers import XGBRelationClassifier
from models.comparators import get_sliced_relation_mention
import argparse
from models.ner import wikifier, CoreferenceResolver
from config import V3_CONFIG
import json
import torch
from models.encoders import PairEncoder
import numpy as np
from benchmark import usa_benchmark
from v2 import train_v2


class V3(object):
    """
    Upon creation, loads a trained XGB model (trained using v3.py using the parameters in the config file).\n
    This model classifies entity pairs (ie a pair of word groups detected by wikifier) into relations. The word pairs
    are encoded using BERT's attentions, and we classify these pair encodings using an XGB model.
    The training procedure uses the configuration in V3_XGB_CONFIG, and this model uses the parameters in V3_CONFIG.
    """
    def __init__(self, experiment_name='v3_trained'):
        self.config = V3_CONFIG
        self.xgb = XGBRelationClassifier(experiment_name, load=True, model_type='v3')
        self.coreference_resolver = CoreferenceResolver()
        self.max_entity_pair_distance = self.config['max_entity_pair_distance']
        self.bilateral_context = self.config['bilateral_context']
        self.threshold = self.config['threshold']
        self.max_sentence_length = self.config['max_sentence_length']

        with open('wikidatavitals/data/encoded/relation_indices.json', 'r') as f:
            self.relations_by_idx = json.load(f)

        self.device = torch.device('cuda:0')
        self.PE = PairEncoder(max_sentence_length=self.max_sentence_length)

    def extract_facts(self, sentence, verbose=False):
        facts = []
        # NER
        processed_text = self.coreference_resolver(sentence)
        wikifier_results = wikifier(processed_text)

        # creating entity pairs: only pairs (e1, e2) in order
        # with at most 'max_entity_pair_distance - 1' entities between them
        n_mentions = len(wikifier_results)

        if n_mentions < 2 and verbose:
            print('Not enough detected entities: ', n_mentions)

        with torch.no_grad():
            for e1_idx in range(n_mentions):
                for e2_idx in range(n_mentions):
                    if e1_idx == e2_idx or abs(e1_idx - e2_idx) > self.max_entity_pair_distance:
                        continue
                    # Preparing XGB input
                    e1_dict, e2_dict = wikifier_results[e1_idx], wikifier_results[e2_idx]
                    sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, processed_text,
                                                                  bilateral_context=self.bilateral_context)

                    if len(sliced_sentence.split(' ')) > self.max_sentence_length:
                        continue

                    e1_slice = e1_dict['start_idx'], e1_dict['end_idx'] + 1
                    e2_slice = e2_dict['start_idx'], e2_dict['end_idx'] + 1
                    xgb_input = self.PE.get_pair_encoding(sliced_sentence.split(' '), e1_slice, e2_slice).cpu().numpy()
                    xgb_input = xgb_input.reshape((1, 288))  # unclear why this is necessary, og shape is (288,)

                    # Computing XGB output
                    probabilities = self.xgb.model.predict_proba(xgb_input)[0]  # shape (50)
                    chosen_idx = np.argmax(probabilities)
                    best_probability = probabilities[chosen_idx]

                    if verbose:
                        print('e1:\tmention={}\tname={}\ne2:\tmention={}\tname={}'.format(
                            e1_dict['mention'], e1_dict['name'], e2_dict['mention'], e2_dict['name']
                        ),
                            'best property_name={}\tid={}\tsim={}\n\n'.format(
                                self.relations_by_idx[chosen_idx]['name'], self.relations_by_idx[chosen_idx]['id'],
                                best_probability
                            )
                        )

                    if best_probability > self.threshold:
                        facts.append({
                            'sentence': sliced_sentence,
                            'e1_mention': e1_dict['mention'],
                            'e1_name': e1_dict['name'],
                            'e1_id': e1_dict['id'],
                            'e2_mention': e2_dict['mention'],
                            'e2_name': e2_dict['name'],
                            'e2_id': e2_dict['id'],
                            'property_name': self.relations_by_idx[chosen_idx]['name'],
                            'property_id': self.relations_by_idx[chosen_idx]['id'],
                            'probability': str(best_probability)
                        })

        return facts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', '--train', action='store_true', help='Trains and saves an XGB classifier ')
    parser.add_argument('--s', '--sentence', type=str, default='', help="Sentence to extract facts from using v3")
    parser.add_argument('--b', '--benchmark', action='store_true', help="Run the USA benchmark")
    args = parser.parse_args()

    if args.s:
        v3 = V3()
        print('Parsing "{}" ...'.format(args.s))
        v3.extract_facts(args.s, verbose=True)

    if args.t:
        print('Training v3 ...')
        train_v2(model_type='v3')

    if args.b:
        v3 = V3()
        usa_benchmark(v3, v3.config)
