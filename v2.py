from models.classifiers import XGBRelationClassifier
from models.comparators import get_sliced_relation_mention
import argparse
from models.ner import wikifier, CoreferenceResolver
from config import V2_CONFIG
import json
from transformers import BertTokenizer, BertModel
import torch
from models.encoders import preprocess_sentences_encoder
import numpy as np
from benchmark import usa_benchmark


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


class V2(object):
    """
    Upon creation, loads a trained XGB model (trained using v2.py using the parameters in the config file).\n
    There are two versions: V2 and V2.5 trained to classify sentences into relations: \n
    - V2: This model is trained using BERT representations on 'sentences' of the form <e1><verb><e2> from Wikidata.
    - V2.5: This model is trained using Wikipedia sentences annotated using known Wikidata facts.
    The model classifies within 50 relations from Wikidata.
    The training procedure uses the configuration in XGB_CONFIG, and this model uses the parameters in V2_CONFIG.
    """
    def __init__(self, experiment_name='trained', model_type='v2'):
        if experiment_name == 'trained':  # handling the two default possibilities
            experiment_name = model_type + '_trained'
        self.xgb = XGBRelationClassifier(experiment_name, load=True, model_type=model_type)
        self.coreference_resolver = CoreferenceResolver()
        self.max_entity_pair_distance = V2_CONFIG['max_entity_pair_distance']
        self.bilateral_context = V2_CONFIG['bilateral_context']
        self.threshold = V2_CONFIG['threshold']
        self.max_sentence_length = V2_CONFIG['max_sentence_length']

        with open('wikidatavitals/data/encoded/relation_indices.json', 'r') as f:
            self.relations_by_idx = json.load(f)

        self.device = torch.device('cuda:0')
        self.bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased").cuda().eval()

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
                for e2_idx in range(e1_idx + 1, min(e1_idx + self.max_entity_pair_distance, n_mentions)):
                    # Preparing XGB input
                    e1_dict, e2_dict = wikifier_results[e1_idx], wikifier_results[e2_idx]
                    sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, processed_text,
                                                                  bilateral_context=self.bilateral_context)

                    if len(sliced_sentence.split(' ')) > self.max_sentence_length:
                        continue

                    ids, masks = preprocess_sentences_encoder(sliced_sentence, self.bert_tokenizer, self.device)
                    model_hidden_states = self.bert(ids, attention_mask=masks).last_hidden_state  # shape (1, 16, 768)
                    model_output = model_hidden_states[:, 0, :]  # use the CLS output: shape (1, 768)
                    xgb_input = model_output.cpu().numpy()

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
    parser.add_argument('--s', '--sentence', type=str, default='', help="Sentence to extract facts from using v2")
    parser.add_argument('--p5', '--point-five', '--point-5', action='store_true', help='Switches to v2.5 instead of v2')
    parser.add_argument('--b', '--benchmark', action='store_true', help="Run the USA benchmark")
    args = parser.parse_args()

    m_type = 'v2.5' if args.p5 else 'v2'

    if args.s:
        v2 = V2(model_type=m_type)
        print('Parsing "{}" ...'.format(args.s))
        v2.extract_facts(args.s, verbose=True)

    if args.t:
        print('Training {} ...'.format(m_type))
        train_v2(model_type=m_type)

    if args.b:
        v2 = V2(model_type=m_type)
        saved_config_for_benchmark = V2_CONFIG.copy()  # this dictionary is saved in the benchmark output for legibility
        saved_config_for_benchmark['model_type'] = m_type
        usa_benchmark(v2, saved_config_for_benchmark)
