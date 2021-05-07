import os.path
import json
from wikidata.client import Client
from wikidata.entity import EntityId
from torchkge.utils.datasets import load_wikidata_vitals
import argparse
from tqdm import tqdm


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


def build_property_verbs_dictionary(n_verbs):
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


def build_entity_dictionary():
    """
    Dumps a dictionary mapping every wikidatavitals-vitals entity ID with their name
    """
    kg, _ = load_wikidata_vitals()

    with open('wikidatavitals/data/entities.json', 'w') as f:
        json.dump(kg.entid2pagename, f, indent=4)


def build_verb_idx_to_relation_list(verbs_file='wikidatavitals/data/property_verbs.json'):

    with open(verbs_file, 'r') as f:
        verbs = json.load(f)

    res = []

    for verb_id, verb_list in verbs.items():
        res.extend([verb_id]*len(verb_list))

    with open('wikidatavitals/data/verb_idx2id.json', 'w') as f:
        json.dump(res, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', '--verbs', action='store_true')
    parser.add_argument('--n-verbs', type=int, default=5, required=False)
    parser.add_argument('--e', '--entities', action='store_true')
    args = parser.parse_args()

    if args.v:
        print('Building the property-to-verbs dictionary ...')
        build_property_verbs_dictionary(args.n_verbs)
        build_verb_idx_to_relation_list()

    if args.e:
        print('Building the entity-to-name dictionary ...')
        build_entity_dictionary()
