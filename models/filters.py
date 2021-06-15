import json
from tqdm import tqdm


def save_entity_types():
    with open('wikidatavitals/data/relations.json', 'r') as f:
        relations = json.load(f)

    entity_to_types_dict = {}

    for (h, r, t) in tqdm(relations):
        if r == 'P31':  # is instance of or rdf:type
            if h not in entity_to_types_dict:
                entity_to_types_dict[h] = []
            entity_to_types_dict[h].append(t)

    with open('wikidatavitals/data/entity_types.json', 'w') as f:
        json.dump(entity_to_types_dict, f)


def save_relation_argument_types():
    with open('wikidatavitals/data/relations.json', 'r') as f:
        relations = json.load(f)

    relation_to_types_tuple_dict = {}  # for each relation key: two lists of possible types (for h and t)

    for (h, r, t) in tqdm(relations):
        if r not in relation_to_types_tuple_dict:
            relation_to_types_tuple_dict[r] = {'h': [], 't': []}
        relation_to_types_tuple_dict[r]['h'].append(h)
        relation_to_types_tuple_dict[r]['t'].append(t)

    with open('wikidatavitals/data/relation_types.json', 'w') as f:
        json.dump(relation_to_types_tuple_dict, f)


class TypeFilter(object):
    def __init__(self):
        with open('wikidatavitals/data/entity_types.json', 'r') as f:
            self.entity_types = json.load(f)
        with open('wikidatavitals/data/relation_types.json', 'r') as f:
            self.relation_types = json.load(f)

    def accept(self, h, r, t):
        return h in self.relation_types[r]['h'] and t in self.relation_types[t]['h']
