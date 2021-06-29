import json
from models.ner import wikifier, CoreferenceResolver
from models.filters import TypeFilter


class NoFact(Exception):
    """Raised internally if an Extractor finds no fact for an entity pair"""
    pass


class Extractor:
    """Base class for all extractor versions: extracts fact triplets from a given sentence"""

    def __init__(self, n_relations=50, max_entity_pair_distance=3, type_filter=True):
        with open('wikidatavitals/data/relation_counts.json', 'r') as f:
            self.relation_counts = json.load(f)

        self.n_relations = n_relations
        self.max_entity_pair_distance = max_entity_pair_distance
        self.relation_ids = [c[0] for c in self.relation_counts[:self.n_relations]]

        self.coreference_resolver = CoreferenceResolver()

        with open('wikidatavitals/data/relation_names.json', 'r') as f:
            self.relation_names = json.load(f)

        self.placeholder_relation_ids = ['r_id placeholder']  # for _get_relation overriding
        self.filter = type_filter
        if type_filter:
            self.TF = TypeFilter()

    def _get_entity_pairs_and_processed_text(self, sentence):
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

        return entity_pairs, processed_text

    def extract_facts(self, sentence, verbose=True):
        entity_pairs, processed_text = self._get_entity_pairs_and_processed_text(sentence)
        facts = []
        for pair_idx in range(len(entity_pairs)):
            e1_dict, e2_dict = entity_pairs[pair_idx]['e1'], entity_pairs[pair_idx]['e2']

            try:
                r_ids = self._get_relation(e1_dict, e2_dict, processed_text)
                r_id = r_ids[0]  # the output is the most likely relation

                if self.filter:
                    if not self.TF.accept(e1_dict['id'], r_id, e2_dict['id']):
                        raise NoFact

                facts.append({
                    'e1_mention': e1_dict['mention'],
                    'e1_name': e1_dict['name'],
                    'e1_id': e1_dict['id'],
                    'e2_mention': e2_dict['mention'],
                    'e2_name': e2_dict['name'],
                    'e2_id': e2_dict['id'],
                    'property_id': r_id,
                    'property_name': self.relation_names[r_id],
                    'ordered_candidates': r_ids
                })

                if verbose:
                    print('e1:\tmention={}\tname={}\ne2:\tmention={}\tname={}'.format(
                        e1_dict['mention'], e1_dict['name'], e2_dict['mention'], e2_dict['name']),
                        'relation name={}\tid={}\n\n'.format(self.relation_names[r_id], r_id))

            except NoFact:
                continue

        return facts

    def _get_relation(self, e1_dict, e2_dict, processed_text):
        """
        Version-dependent routine that will try to find an ordered list of candidate relations for the given entities
        and text \n
        :return the estimated ids: a list of strings (relation ids) from most likely to least likely
        """
        return self.placeholder_relation_ids  # this function needs to return something hence this return
