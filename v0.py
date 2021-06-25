import json
import argparse
from benchmark import usa_benchmark, hundo_benchmark
from config import V0_CONFIG
from models.filters import has_intersection
from extractor_base import Extractor, NoFact


class V0(Extractor):
    """
    The configuration for this model is in 'config.py'\n
    Implements the V0 model:
    Given a sentence, we detect entities using the wikifier algorithm, then for each pair we output the the most popular
    relation that has legal types. A triplet is considered legal if the types of the entities
    (as given) by the instance_of relation in Wikidata have already been in Wikidata with the current relation.
    """

    def __init__(self):
        n_relations, max_entity_pair_distance = V0_CONFIG['n_relations'], V0_CONFIG['max_entity_pair_distance']
        super().__init__(n_relations=n_relations, max_entity_pair_distance=max_entity_pair_distance)

        with open('wikidatavitals/data/entity_types.json', 'r') as f:
            self.entity_types = json.load(f)

        with open('wikidatavitals/data/relation_types.json', 'r') as f:
            self.relation_types = json.load(f)

    def _get_relation(self, e1_dict, e2_dict, processed_text):
        e1, e2 = e1_dict['id'], e2_dict['id']

        try:
            e1_types, e2_types = self.entity_types[e1], self.entity_types[e2]  # list of possible types
        except KeyError:  # this happens if the wikifier finds entities that are not in Wikidata-vitals -> skip
            raise NoFact

        legal_relations = [r for r, d in self.relation_types.items()
                           if has_intersection(e1_types, d['h']) and has_intersection(e2_types, d['t'])]

        legal_relations_ordered = [r for (r, _) in self.relation_counts
                                   if r in legal_relations and r in self.relation_ids]

        if legal_relations_ordered:  # if the list is empty it means that no legal fact was found.
            return legal_relations_ordered[0]  # most popular

        raise NoFact


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--s', '--sentence', type=str, default='', help="Sentence to extract facts from using v0")
    parser.add_argument('--u-b', '--usa-benchmark', action='store_true', help="Run the USA benchmark")
    parser.add_argument('--h-b', '--hundo-benchmark', action='store_true', help="Run the Hundo benchmark")
    args = parser.parse_args()

    v0 = V0()

    if args.s:
        print('Parsing "{}" ...'.format(args.sentence))
        v0.extract_facts(args.sentence, verbose=True)

    if args.u_b:
        usa_benchmark(v0, V0_CONFIG)

    if args.h_b:
        hundo_benchmark(v0, V0_CONFIG)
