from models.comparators import UniversalSentenceEncoderComparator, get_comparison_sentences, get_sliced_relation_mention
import json
import numpy as np
import argparse
from benchmark import usa_benchmark, hundo_benchmark, simple_benchmark
from config import V1_CONFIG
from extractor_base import Extractor, NoFact


def get_unique(list_with_repetition):
    """
    This is a deterministic equivalent to list(set()) which conserves order
    :param list_with_repetition: a list of objects with potential repetitions
    :return: a list containing each element of the input only once in the order of first appearance
    """

    res = []
    for x in list_with_repetition:
        if x not in res:
            res.append(x)
    return res


class V1(Extractor):
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
        n_relations, max_entity_pair_distance = V1_CONFIG['n_relations'], V1_CONFIG['max_entity_pair_distance']
        super().__init__(n_relations=n_relations, max_entity_pair_distance=max_entity_pair_distance)
        self.comparator = UniversalSentenceEncoderComparator()
        self.threshold = V1_CONFIG['threshold']
        self.double_check = V1_CONFIG['double_check']
        self.bilateral_context = V1_CONFIG['bilateral_context']

        with open('wikidatavitals/data/property_verbs.json', 'r') as f:
            all_verbs = json.load(f)

        with open('wikidatavitals/data/verb_idx2id.json', 'r') as f:
            all_verb_idx2id = json.load(f)

        self.verbs = {relation_id: verb_list for relation_id, verb_list in all_verbs.items()
                      if relation_id in self.relation_ids}

        self.verb_idx2id = [property_id for property_id in all_verb_idx2id if property_id in self.relation_ids]

    def _get_relation(self, e1_dict, e2_dict, processed_text):
        comparison_sentences = get_comparison_sentences(e1_dict['mention'],
                                                        e2_dict['mention'],
                                                        self.verbs,
                                                        double_check=self.double_check)
        sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, processed_text,
                                                      bilateral_context=self.bilateral_context)
        sims = self.comparator.compare_sentences(sliced_sentence, comparison_sentences)
        chosen_relation_idx = np.argmax(sims)

        if self.double_check:  # censor the fact if it doesn't pass the double check test
            if chosen_relation_idx == len(comparison_sentences) - 1:  # last sentence is <e1> <e2> here
                raise NoFact

        # slice :50 for the double_check option which adds an additional test with no relation -> 51 instead of 50
        sorted_sims = np.sort(
            np.array(
                [(self.verb_idx2id[idx], sim) for idx, sim in enumerate(sims[:50])],
                dtype=[('r_id', 'U10'), ('sim', float)]
            ),
            order='sim')

        # filtering on the output (most similar relation)
        if sorted_sims['sim'][-1] > self.threshold:
            return get_unique(sorted_sims['r_id'][::-1])

        raise NoFact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', '--sentence', type=str, default='', help="Sentence to extract facts from using v1")
    parser.add_argument('--u-b', '--usa-benchmark', action='store_true', help="Run the USA benchmark")
    parser.add_argument('--h-b', '--hundo-benchmark', action='store_true', help="Run the Hundo benchmark")
    parser.add_argument('--s-b', '--simple-benchmark', action='store_true', help="Run the Simple benchmark")
    args = parser.parse_args()

    v1 = V1()

    if args.s:
        print('Parsing "{}" ...'.format(args.sentence))
        v1.extract_facts(args.sentence, verbose=True)

    if args.u_b:
        usa_benchmark(v1, V1_CONFIG)

    if args.h_b:
        hundo_benchmark(v1, V1_CONFIG)

    if args.s_b:
        simple_benchmark(v1, V1_CONFIG)
