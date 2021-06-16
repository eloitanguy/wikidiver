import json
from tqdm import tqdm


def has_intersection(list1, list2):
    return bool(list(set(list1) & set(list2)))


def union_without_repetition(list1, list2):
    return list(set(list1) | set(list2))


def save_entity_types():
    with open('wikidatavitals/data/relations.json', 'r') as f:
        relations = json.load(f)

    entity_to_types_dict = {}

    # initialise all entities a having for type possibilities []
    # -> ensures that all entities in wikidata-vitals have an entry
    for (h, _, t) in relations:
        entity_to_types_dict[h] = []
        entity_to_types_dict[t] = []

    for (h, r, t) in tqdm(relations):
        if r == 'P31':  # is instance of or rdf:type
            entity_to_types_dict[h].append(t)

    with open('wikidatavitals/data/entity_types.json', 'w') as f:
        json.dump(entity_to_types_dict, f)


def save_relation_argument_types():
    with open('wikidatavitals/data/relations.json', 'r') as f:
        relations = json.load(f)

    with open('wikidatavitals/data/entity_types.json', 'r') as f:
        entity_types = json.load(f)

    relation_to_types_tuple_dict = {}  # for each relation key: two lists of possible types (for h and t)

    for (h, r, t) in tqdm(relations):
        if r not in relation_to_types_tuple_dict:
            relation_to_types_tuple_dict[r] = {'h': [], 't': []}
        relation_to_types_tuple_dict[r]['h'] = union_without_repetition(entity_types[h],
                                                                        relation_to_types_tuple_dict[r]['h'])
        relation_to_types_tuple_dict[r]['t'] = union_without_repetition(entity_types[t],
                                                                        relation_to_types_tuple_dict[r]['t'])

    with open('wikidatavitals/data/relation_types.json', 'w') as f:
        json.dump(relation_to_types_tuple_dict, f)


class TypeFilter(object):
    def __init__(self):
        with open('wikidatavitals/data/entity_types.json', 'r') as f:
            self.entity_types = json.load(f)
        with open('wikidatavitals/data/relation_types.json', 'r') as f:
            self.relation_types = json.load(f)

    def accept(self, h, r, t):
        accept_h = h in self.relation_types[r]['h'] or self.relation_types[r]['h'] == []
        accept_t = t in self.relation_types[r]['t'] or self.relation_types[r]['t'] == []
        return accept_h and accept_t
