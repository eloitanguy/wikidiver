from models.ner import wikifier, CoreferenceResolver
import json
import argparse
from benchmark import usa_benchmark
from config import V0_CONFIG
from models.filters import has_intersection, union_without_repetition


class V0(object):
    """
    The configuration for this model is in 'config.py'\n
    Implements the V0 model:
    Given a sentence, we detect entities using the wikifier algorithm, then for each pair we output the the most popular
    relation that has legal types. A triplet is considered legal if the types of the entities
    (as given) by the instance_of relation in Wikidata have already been in Wikidata with the current relation.
    """

    def __init__(self):
        with open('wikidatavitals/data/relation_counts.json', 'r') as f:
            self.relation_counts = json.load(f)

        self.n_relations = V0_CONFIG['n_relations']
        self.max_entity_pair_distance = V0_CONFIG['max_entity_pair_distance']
        self.relation_ids = [c[0] for c in self.relation_counts[:self.n_relations]]

        with open('wikidatavitals/data/entity_types.json', 'r') as f:
            self.entity_types = json.load(f)

        with open('wikidatavitals/data/relation_types.json', 'r') as f:
            self.relation_types = json.load(f)

        with open('wikidatavitals/data/relation_names.json', 'r') as f:
            self.relation_names = json.load(f)

        self.coreference_resolver = CoreferenceResolver()

    def extract_facts(self, sentence, verbose=False):
        facts = []

        # Step 1: NER
        processed_text = self.coreference_resolver(sentence)
        wikifier_results = wikifier(processed_text)

        # creating entity pairs: only pairs (e1, e2) in order
        # with at most 'max_entity_pair_distance - 1' entities between them
        entity_pairs = []
        n_mentions = len(wikifier_results)
        for e1_idx in range(n_mentions):
            for e2_idx in range(e1_idx + 1, min(e1_idx + self.max_entity_pair_distance, n_mentions)):
                entity_pairs.append({'e1': wikifier_results[e1_idx],
                                     'e2': wikifier_results[e2_idx]})

        # Finding a relation for each pair
        for pair_idx in range(len(entity_pairs)):
            e1_dict, e2_dict = entity_pairs[pair_idx]['e1'], entity_pairs[pair_idx]['e2']
            e1, e2 = e1_dict['id'], e2_dict['id']

            try:
                e1_types, e2_types = self.entity_types[e1], self.entity_types[e2]  # list of possible types
            except KeyError:  # this happens if the wikifier finds entities that are not in Wikidata-vitals -> skip
                continue

            legal_relations = [r for r, d in self.relation_types.items()
                               if has_intersection(e1_types, d['h']) and has_intersection(e2_types, d['t'])]

            legal_relations_ordered = [r for (r, _) in self.relation_counts
                                       if r in legal_relations and r in self.relation_ids]

            if legal_relations_ordered:  # if the list is empty it means that no legal fact was found.
                chosen_ID = legal_relations_ordered[0]  # most popular
                if verbose:
                    print('e1:\tmention={}\tname={}\ne2:\tmention={}\tname={}'.format(
                        e1_dict['mention'], e1_dict['name'], e2_dict['mention'], e2_dict['name']),
                        'relation name={}\tid={}\n\n'.format(self.relation_names[chosen_ID], chosen_ID))
                facts.append({
                    'e1_mention': e1_dict['mention'],
                    'e1_name': e1_dict['name'],
                    'e1_id': e1_dict['id'],
                    'e2_mention': e2_dict['mention'],
                    'e2_name': e2_dict['name'],
                    'e2_id': e2_dict['id'],
                    'property_id': chosen_ID,
                    'property_name': self.relation_names[chosen_ID],
                })

        return facts


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', '--sentence', type=str, default='', help="Sentence to extract facts from using v1")
    parser.add_argument('--b', '--benchmark', action='store_true', help="Run the USA benchmark")
    args = parser.parse_args()

    v0 = V0()

    if args.s:
        print('Parsing "{}" ...'.format(args.sentence))
        v0.extract_facts(args.sentence, verbose=True)

    if args.b:
        usa_benchmark(v0, V0_CONFIG)
