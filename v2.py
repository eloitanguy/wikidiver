from models.classifiers import XGBRelationClassifier
from models.comparators import get_sliced_relation_mention
import argparse
from config import V2_CONFIG, V2p5_CONFIG
import json
from transformers import BertTokenizer, BertModel
import torch
from models.encoders import preprocess_sentences_encoder
import numpy as np
from benchmark import usa_benchmark, hundo_benchmark, simple_benchmark
from extractor_base import Extractor, NoFact


def train_v2(model_type='v2'):
    """
    Trains and saves an XGB classifier using the configuration in the config file.\n
    This requires the encoded Wikidata dataset, obtainable using 'python wikidatavitals/dataset --encode'.\n
    Training takes around 30 minutes.
    """
    experiment_name = model_type + '_trained'
    x = XGBRelationClassifier(experiment_name, model_type=model_type)
    x.train()
    x.val()
    x.save()


class V2(Extractor):
    """
    Upon creation, loads a trained XGB model (trained using v2.py using the parameters in the config file).\n
    There are two versions: V2 and V2.5 trained to classify sentences into relations: \n
    - V2: This model is trained using BERT representations on 'sentences' of the form <e1><verb><e2> from Wikidata.
    - V2.5: This model is trained using Wikipedia sentences annotated using known Wikidata facts.
    The model classifies within 50 relations from Wikidata.
    The training procedure uses the configuration in V2(p5)XGB_CONFIG, and this model uses the parameters in V2_CONFIG.
    """
    def __init__(self, experiment_name='trained', model_type='v2'):
        self.config = V2_CONFIG if model_type == 'v2' else V2p5_CONFIG
        max_entity_pair_distance = self.config['max_entity_pair_distance']
        super().__init__(max_entity_pair_distance=max_entity_pair_distance)  # n_relations is imposed at 50 here

        if experiment_name == 'trained':  # handling the two default possibilities
            experiment_name = model_type + '_trained'

        self.xgb = XGBRelationClassifier(experiment_name, load=True, model_type=model_type)
        self.bilateral_context = self.config['bilateral_context']
        self.threshold = self.config['threshold']
        self.max_sentence_length = self.config['max_sentence_length']

        with open('wikidatavitals/data/encoded/relation_indices.json', 'r') as f:
            self.relations_by_idx = json.load(f)

        self.device = torch.device('cuda:0')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased").cuda().eval()

    def _get_relation(self, e1_dict, e2_dict, processed_text):
        with torch.no_grad():
            sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, processed_text,
                                                          bilateral_context=self.bilateral_context)

            if len(sliced_sentence.split(' ')) > self.max_sentence_length:
                raise NoFact

            ids, masks = preprocess_sentences_encoder(sliced_sentence, self.bert_tokenizer, self.device)
            model_hidden_states = self.bert(ids, attention_mask=masks).last_hidden_state  # shape (1, 16, 768)
            model_output = model_hidden_states[:, 0, :]  # use the CLS output: shape (1, 768)
            xgb_input = model_output.cpu().numpy()

            # Computing XGB output
            probabilities = self.xgb.model.predict_proba(xgb_input)[0]  # shape (50)

            sorted_probabilities = np.sort(
                np.array(
                    [(self.relations_by_idx[idx]['id'], p) for idx, p in enumerate(probabilities)],
                    dtype=[('r_id', 'U10'), ('p', float)]
                ),
                order='p')

            if sorted_probabilities['p'][-1] > 0:#self.threshold:
                return sorted_probabilities['r_id'][::-1].tolist()

            raise NoFact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--t', '--train', action='store_true', help='Trains and saves an XGB classifier ')
    parser.add_argument('--s', '--sentence', type=str, default='', help="Sentence to extract facts from using v2")
    parser.add_argument('--p5', '--point-five', '--point-5', action='store_true', help='Switches to v2.5 instead of v2')
    parser.add_argument('--u-b', '--usa-benchmark', action='store_true', help="Run the USA benchmark")
    parser.add_argument('--h-b', '--hundo-benchmark', action='store_true', help="Run the Hundo benchmark")
    parser.add_argument('--s-b', '--simple-benchmark', action='store_true', help="Run the Simple benchmark")
    args = parser.parse_args()

    m_type = 'v2.5' if args.p5 else 'v2'

    if args.t:
        print('Training {} ...'.format(m_type))
        train_v2(model_type=m_type)

    else:
        v2 = V2(model_type=m_type)
        if args.s:
            print('Parsing "{}" ...'.format(args.s))
            v2.extract_facts(args.s, verbose=True)

        if args.u_b:
            usa_benchmark(v2, v2.config)

        if args.h_b:
            hundo_benchmark(v2, v2.config)

        if args.s_b:
            simple_benchmark(v2, v2.config)
