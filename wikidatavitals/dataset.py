import os.path
import json
from wikidata.client import Client
from wikidata.entity import EntityId
from torchkge.utils.datasets import load_wikidata_vitals
import argparse
from tqdm import tqdm, trange
import numpy as np


def get_names(ID: str, number, client=None):
    """
    Returns the name and aliases of a given wikidatavitals ID,
    Used for properties in order to get different verbs
    """
    if client is None:  # this avoids creating a client at every request
        client = Client()

    entity_id = EntityId(ID)
    entity = client.get(entity_id, load=True)
    names = [str(entity.label)]

    try:
        aliases = entity.attributes['aliases']['en']
        for idx, entry_dict in enumerate(aliases):
            if idx + 1 >= number:
                break
            names.append(entry_dict['value'])
    except KeyError:  # there might be no aliases, in this case we just output the name
        pass

    return names


def save_property_verbs_dictionary(n_verbs):
    """
    Dumps a dictionary mapping property wikidatavitals IDs to a list of verbs that represent them
    to wikidatavitals/data/property_verbs.json
    """
    kg, _ = load_wikidata_vitals()
    property_ids = list(kg.relid2label.keys())
    client = Client()
    property_verbs_dictionary = {}

    for ID in tqdm(property_ids):
        names = get_names(ID, n_verbs, client=client)
        property_verbs_dictionary[ID] = names

    if not os.path.exists('wikidatavitals/data/'):
        os.makedirs('wikidatavitals/data/')

    with open('wikidatavitals/data/property_verbs.json', 'w') as f:
        json.dump(property_verbs_dictionary, f, indent=4)


def save_entity_dictionary():
    """
    Dumps a dictionary mapping every wikidatavitals-vitals entity ID with their name
    """
    kg, _ = load_wikidata_vitals()

    if not os.path.exists('wikidatavitals/data/'):
        os.makedirs('wikidatavitals/data/')

    with open('wikidatavitals/data/entity_names.json', 'w') as f:
        json.dump(kg.entid2pagename, f, indent=4)

    client = Client()
    entity_aliases = {}

    for ID in tqdm(list(kg.entid2pagename.keys())):
        entity_aliases[ID] = get_names(ID, 10, client)

    with open('wikidatavitals/data/entity_aliases.json', 'w') as f:
        json.dump(entity_aliases, f, indent=4)


def save_verb_idx_to_relation_list(verbs_file='wikidatavitals/data/property_verbs.json'):
    with open(verbs_file, 'r') as f:
        verbs = json.load(f)

    res = []

    for verb_id, verb_list in verbs.items():
        res.extend([verb_id] * len(verb_list))

    if not os.path.exists('wikidatavitals/data/'):
        os.makedirs('wikidatavitals/data/')

    with open('wikidatavitals/data/verb_idx2id.json', 'w') as f:
        json.dump(res, f, indent=4)


def save_relations():
    kg, _ = load_wikidata_vitals()
    heads_idx, relations_idx, tail_idx = kg.head_idx, kg.relations, kg.tail_idx
    triplets = []
    n_rel = np.shape(relations_idx)[0]
    entity_idx_to_id = {v: k for k, v in kg.ent2ix.items()}
    relation_idx_to_id = {v: k for k, v in kg.rel2ix.items()}

    for rel_idx in trange(n_rel):
        head_id = entity_idx_to_id[heads_idx[rel_idx].item()]
        relation_id = relation_idx_to_id[relations_idx[rel_idx].item()]
        tail_id = entity_idx_to_id[tail_idx[rel_idx].item()]
        triplets.append([head_id, relation_id, tail_id])

    if not os.path.exists('wikidatavitals/data/'):
        os.makedirs('wikidatavitals/data/')

    with open('wikidatavitals/data/relations.json', 'w') as f:
        json.dump(triplets, f)  # the file is large and 'vertical' so we put no indent

    with open('wikidatavitals/data/relation_names.json', 'w') as f:
        json.dump(kg.relid2label, f, indent=4)

    # computing relation counts
    count_dict = {}
    for triplet in triplets:
        relation = triplet[1]
        if relation not in count_dict:
            count_dict[relation] = 0
        count_dict[relation] += 1

    values = [(ID, count) for (ID, count) in count_dict.items()]
    sorted_relations = np.sort(np.array(values, dtype=[('id', 'S10'), ('count', int)]), order='count')[::-1]
    sorted_relations_list = [(str(ID).replace("'", '').replace('b', ''), str(count)) for (ID, count) in sorted_relations.tolist()]

    with open('wikidatavitals/data/relation_counts.json', 'w') as f:
        json.dump(sorted_relations_list, f, indent=4)


# class WikiDataVitalsSentences(Dataset):
#     def __init__(self, dataset_type):
#         assert dataset_type in ["train", "val"]
#
#         with open('wikidatavitals/data/relations.json', 'r') as f:
#             all_relations = json.load(f)
#
#         train_val_split = int(0.66*len(all_relations))
#         self.triplets = all_relations[:train_val_split] if dataset_type == 'train' else all_relations[train_val_split:]
#
#         with open('wikidatavitals/data/property_verbs.json', 'r') as f:
#             self.verbs = json.load(f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', '--verbs', action='store_true')
    parser.add_argument('--n-verbs', type=int, default=5, required=False)
    parser.add_argument('--e', '--entities', action='store_true')
    parser.add_argument('--r', '--relations', action='store_true')
    args = parser.parse_args()

    if args.v:
        print('Building the property-to-verbs dictionary ...')
        save_property_verbs_dictionary(args.n_verbs)
        save_verb_idx_to_relation_list()

    if args.e:
        print('Building the entity-to-name dictionary ...')
        save_entity_dictionary()

    if args.r:
        print('Building the relation triplet list ...')
        save_relations()
