import os.path
import json
from wikidata.client import Client
from wikidata.entity import EntityId
from torchkge.utils.datasets import load_wikidata_vitals
from tqdm import tqdm, trange
import numpy as np
from torch.utils.data import Dataset
from random import Random


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
    sorted_relations_list = [(str(ID).replace("'", '').replace('b', ''), str(count)) for (ID, count) in
                             sorted_relations.tolist()]

    with open('wikidatavitals/data/relation_counts.json', 'w') as f:
        json.dump(sorted_relations_list, f, indent=4)


class WikiDataVitalsSentences(Dataset):
    """
    A torch (string) Dataset containing pseudo-sentences of the form <e1><r><e2> generated from Wikidata-vitals.\n
    Each entry is a dictionary:\n
    {\t'sentence': [a pseudo-sentence],
    \t'label': [the idx of the relation]}
    """

    def __init__(self, dataset_type, n_relations=50):
        assert dataset_type in ["train", "val"]
        np.random.seed(42)  # ensures reproducibility
        self.n_relations = n_relations

        with open('wikidatavitals/data/relations.json', 'r') as f:
            all_relations = json.load(f)  # loading all the relation triplets in wikidata-vitals
        Random(42).shuffle(all_relations)  # shuffle relations in place with a set seed

        with open('wikidatavitals/data/relation_counts.json', 'r') as f:
            relation_counts = json.load(f)  # loading the ordered relation counts

        with open('wikidatavitals/data/relation_names.json', 'r') as f:
            relation_names = json.load(f)

        self.relation_ids = [c[0] for c in relation_counts[:self.n_relations]]
        self.relation_idx_to_name = [{'id': ID, 'name': relation_names[ID]} for ID in self.relation_ids]
        self.relation_id_to_idx = {ID: idx for idx, ID in enumerate(self.relation_ids)}

        train_val_split = int(0.66 * len(all_relations))
        triplets = all_relations[:train_val_split] if dataset_type == 'train' else all_relations[train_val_split:]
        self.triplets = [t for t in triplets if t[1] in self.relation_ids]  # filtering the top 'n_relations'

        with open('wikidatavitals/data/property_verbs.json', 'r') as f:
            verbs = json.load(f)  # loading the property aliases then filtering the relations
        self.verbs = {property_id: v for property_id, v in verbs.items() if property_id in self.relation_ids}

        with open('wikidatavitals/data/entity_aliases.json', 'r') as f:
            self.entity_aliases = json.load(f)  # loading the entity aliases

        self.n_triplets = len(self.triplets)
        self.n_entities = len(self.entity_aliases.keys())

    def __len__(self):
        return self.n_triplets

    def __getitem__(self, item):
        # outputs a random sentence from the dataset, this is NOT deterministic
        selected_fact_idx = np.random.randint(low=0, high=self.n_triplets)
        e1_id, r_id, e2_id = self.triplets[selected_fact_idx]
        selected_e1_alias_idx = np.random.randint(low=0, high=len(self.entity_aliases[e1_id]))
        selected_r_verb_idx = np.random.randint(low=0, high=len(self.verbs[r_id]))
        selected_e2_alias_idx = np.random.randint(low=0, high=len(self.entity_aliases[e2_id]))

        return {
            'sentence': ' '.join([self.entity_aliases[e1_id][selected_e1_alias_idx],
                                  self.verbs[r_id][selected_r_verb_idx],
                                  self.entity_aliases[e2_id][selected_e2_alias_idx]]),
            'label': self.relation_id_to_idx[r_id]
        }


class FactNotFoundError(Exception):
    """Raised when a FactFinder found no facts about the entity pair"""
    pass


class FactFinder(object):
    def __init__(self):
        with open('wikidatavitals/data/relations.json', 'r') as f:
            self.relations = json.load(f)

    def get_fact(self, e1_id, e2_id):
        """
        Given a pair of entity IDs, scans the wikidatavitals KB for a triplet (e1, r, e2).\n
        If one or more triplets are found, will return the first relation ID that was found.\n
        If no triplet are found, raises FactNotFoundError.
        """
        relation_subset = [(e1, r, e2) for (e1, r, e2) in self.relations if e1 == e1_id and e2 == e2_id]
        if len(relation_subset) > 0:
            return relation_subset[0]
        else:
            raise FactNotFoundError
