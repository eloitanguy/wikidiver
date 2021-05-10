from wikidatavitals.ner import wikifier, CoreferenceResolver
from models.comparators import UniversalSentenceEncoderComparator, get_comparison_sentences, get_sliced_relation_mention
import json
import numpy as np
import argparse
from benchmark import v1_usa_benchmark


class V1(object):
    def __init__(self):
        self.coreference_solver = CoreferenceResolver()
        self.comparator = UniversalSentenceEncoderComparator()

        with open('wikidatavitals/data/property_verbs.json', 'r') as f:
            self.verbs = json.load(f)
        with open('wikidatavitals/data/verb_idx2id.json', 'r') as f:
            self.verb_idx2id = json.load(f)

    def extract_facts(self, sentence, threshold=0.9, max_entity_pair_distance=3, verbose=False):
        facts = []

        # Step 1: NER
        processed_text = self.coreference_solver(sentence)
        wikifier_results = wikifier(processed_text)

        # creating entity pairs: only pairs (e1, e2) in order
        # with at most 'max_entity_pair_distance - 1' entities between them
        entity_pairs = []
        n_mentions = len(wikifier_results)
        for e1_idx in range(n_mentions):
            for e2_idx in range(e1_idx + 1, min(e1_idx + max_entity_pair_distance, n_mentions)):
                entity_pairs.append({'e1': wikifier_results[e1_idx],
                                     'e2': wikifier_results[e2_idx]})

        # Step 2: comparison with generated sentences
        comparison_sentences_by_pair = [get_comparison_sentences(pair_dict['e1']['mention'],
                                                                 pair_dict['e2']['mention'], self.verbs)
                                        for pair_dict in entity_pairs]

        # Comparison of each pair
        for pair_idx in range(len(entity_pairs)):
            comparison_sentences = comparison_sentences_by_pair[pair_idx]
            e1_dict, e2_dict = entity_pairs[pair_idx]['e1'], entity_pairs[pair_idx]['e2']
            sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, processed_text)
            sims = self.comparator.compare_sentences(sliced_sentence, comparison_sentences)
            chosen_relation_idx = np.argmax(sims)
            sim = sims[chosen_relation_idx]

            if verbose:
                print('e1:\tmention={}\tname={}\ne2:\tmention={}\tname={}'.format(
                    e1_dict['mention'], e1_dict['name'], e2_dict['mention'], e2_dict['name']
                ),
                    'best sentence={}\tid={}\tsim={}\n\n'.format(
                        comparison_sentences[chosen_relation_idx], self.verb_idx2id[chosen_relation_idx], sim
                    )
                )

            if sim > threshold:
                facts.append({
                    'e1_mention': e1_dict['mention'],
                    'e1_name': e1_dict['name'],
                    'e1_id': e1_dict['id'],
                    'e2_mention': e2_dict['mention'],
                    'e2_name': e2_dict['name'],
                    'e2_id': e2_dict['id'],
                    'best_sentence': comparison_sentences[chosen_relation_idx],
                    'property_id': self.verb_idx2id[chosen_relation_idx],
                    'sim': str(sim)
                })

        return facts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', '--sentence', type=str, default='', help="Sentence to extract facts from using v1")
    parser.add_argument('--b', '--benchmark', action='store_true', help="Run the USA benchmark")
    args = parser.parse_args()

    v1 = V1()

    if args.s:
        print('Parsing "{}" ...'.format(args.sentence))
        v1.extract_facts(args.sentence, verbose=True)

    if args.b:
        v1_usa_benchmark(v1)
