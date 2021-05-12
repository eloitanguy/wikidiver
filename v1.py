from wikidatavitals.ner import wikifier, CoreferenceResolver
from models.comparators import UniversalSentenceEncoderComparator, get_comparison_sentences, get_sliced_relation_mention
import json
import numpy as np
import argparse
from benchmark import v1_usa_benchmark
from config import V1_CONFIG


class V1(object):
    """
    The configuration for this model is in 'config.py'\n
    Implements the V1 model (given a sentence with two detected entities e1 and e2, compare the slice [e1 ... e2]
    with reference relations built in the form <e1> <verb> <e2>. If the similarity is above 'threshold', we accept the
    fact. \n
    'n_relations' governs how many total relations we can try to extract.\n
    'double_check' indicates whether or not to additionally compare the slice [e1 ... e2] with <e1> <e2>. if this
    similarity is higher than the similarity between [e1 ... e2] with <e1> <verb> <e2>, then we discard the fact. In
    this case we assign a similarity of -1, conveying that the similarity method would not yield good results.
    """
    def __init__(self):
        self.coreference_solver = CoreferenceResolver()
        self.comparator = UniversalSentenceEncoderComparator()
        self.n_relations = V1_CONFIG['n_relations']
        self.threshold = V1_CONFIG['threshold']
        self.double_check = V1_CONFIG['double_check']
        self.bilateral_context = V1_CONFIG['bilateral_context']

        with open('wikidatavitals/data/relation_counts.json', 'r') as f:
            relation_counts = json.load(f)

        self.relations_ids = [c[0] for c in relation_counts[:self.n_relations]]

        with open('wikidatavitals/data/property_verbs.json', 'r') as f:
            all_verbs = json.load(f)

        with open('wikidatavitals/data/verb_idx2id.json', 'r') as f:
            all_verb_idx2id = json.load(f)

        self.verbs = {relation_id: verb_list for relation_id, verb_list in all_verbs.items()
                      if relation_id in self.relations_ids}

        self.verb_idx2id = [property_id for property_id in all_verb_idx2id if property_id in self.relations_ids]

    def extract_facts(self, sentence, max_entity_pair_distance=3, verbose=False):
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
        # if double_check add a comparison sim([e1 ... e2], <e1> <e2) at the end of the sentence list.
        comparison_sentences_by_pair = [get_comparison_sentences(pair_dict['e1']['mention'],
                                                                 pair_dict['e2']['mention'],
                                                                 self.verbs,
                                                                 double_check=self.double_check)
                                        for pair_dict in entity_pairs]

        # Comparison of each pair
        for pair_idx in range(len(entity_pairs)):
            comparison_sentences = comparison_sentences_by_pair[pair_idx]
            e1_dict, e2_dict = entity_pairs[pair_idx]['e1'], entity_pairs[pair_idx]['e2']
            sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, processed_text,
                                                          bilateral_context=self.bilateral_context)
            sims = self.comparator.compare_sentences(sliced_sentence, comparison_sentences)
            chosen_relation_idx = np.argmax(sims)
            sim = sims[chosen_relation_idx]

            if self.double_check:  # censor the fact if it doesn't pass the double check test
                if chosen_relation_idx == len(comparison_sentences) - 1:  # last sentence is <e1> <e2> here
                    sim = -1

            chosen_ID = self.verb_idx2id[chosen_relation_idx] if sim != -1 else 'NONE'
            if verbose:
                print('e1:\tmention={}\tname={}\ne2:\tmention={}\tname={}'.format(
                    e1_dict['mention'], e1_dict['name'], e2_dict['mention'], e2_dict['name']
                ),
                    'best sentence={}\tid={}\tsim={}\n\n'.format(
                        comparison_sentences[chosen_relation_idx], chosen_ID, sim
                    )
                )

            if sim > self.threshold:
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
