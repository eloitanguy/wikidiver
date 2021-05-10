from wikidatavitals.ner import wikifier, CoreferenceResolver
from models.comparators import UniversalSentenceEncoderComparator, get_comparison_sentences
import json
import numpy as np
import argparse


def extract_facts(sentence):
    """
    A test version of V1 on one sentence
    """

    # Step 1: NER
    coreference_solver = CoreferenceResolver()
    processed_text = coreference_solver(sentence)
    wikifier_results = wikifier(processed_text)

    # creating entity pairs (all possibilities, in order)
    entity_pairs = []
    n_mentions = len(wikifier_results)
    for e1_idx in range(n_mentions):
        for e2_idx in range(e1_idx+1, n_mentions):
            entity_pairs.append({'e1': wikifier_results[e1_idx],
                                 'e2': wikifier_results[e2_idx]})

    # Step 2: comparison with generated sentences
    with open('wikidatavitals/data/property_verbs.json', 'r') as f:
        verbs = json.load(f)
    with open('wikidatavitals/data/verb_idx2id.json', 'r') as f:
        verb_idx2id = json.load(f)
    comparison_sentences_by_pair = [get_comparison_sentences(pair_dict['e1']['mention'],
                                                             pair_dict['e2']['mention'], verbs)
                                    for pair_dict in entity_pairs]
    Comparator = UniversalSentenceEncoderComparator()

    # Comparison of each pair
    for pair_idx in range(len(entity_pairs)):
        comparison_sentences = comparison_sentences_by_pair[pair_idx]
        e1_dict, e2_dict = entity_pairs[pair_idx]['e1'], entity_pairs[pair_idx]['e2']
        sims = Comparator.compare_sentences(processed_text, comparison_sentences)
        chosen_relation_idx = np.argmax(sims)

        print('e1:\tmention={}\tname={}\ne2:\tmention={}\tname={}\nbest sentence={}\tid={}\n\n'.format(
            e1_dict['mention'], e1_dict['name'], e2_dict['mention'], e2_dict['name'],
            comparison_sentences[chosen_relation_idx], verb_idx2id[chosen_relation_idx]
        ))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("sentence", help="Sentence to extract facts from using v1")
    args = parser.parse_args()
    print('Parsing "{}" ...'.format(args.sentence))
    extract_facts(args.sentence)
