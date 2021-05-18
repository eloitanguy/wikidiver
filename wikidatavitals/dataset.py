import os.path
import json
from wikidata.client import Client
from wikidata.entity import EntityId
from torchkge.utils.datasets import load_wikidata_vitals
import argparse
from tqdm import tqdm, trange
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import torch
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
        self.n_sentences = self._compute_n_sentences()  # the total amount of possible sentences

    def _compute_n_sentences(self):
        total = 0
        for e1, r, e2 in self.triplets:
            total += len(self.entity_aliases[e1]) * len(self.verbs[r]) * len(self.entity_aliases[e2])

        return total

    def __len__(self):
        return self.n_sentences

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


def save_encoded_WDV_sentences():
    """
    Feeds WikiDataVitalsSentences to BERT-base and saves a numpy .npy file for the train and val sets
    """

    device = torch.device('cuda:0')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained("bert-base-uncased").cuda().eval()

    if not os.path.exists('wikidatavitals/data/encoded/'):
        os.makedirs('wikidatavitals/data/encoded/')

    def preprocess_sentences_encoder(sentences, tokenizer):
        encoded_dict = tokenizer(sentences, add_special_tokens=True, max_length=16, padding='max_length',
                                 truncation=True, return_attention_mask=True,
                                 return_tensors='pt')

        return encoded_dict['input_ids'].to(device), encoded_dict['attention_mask'].to(device)

    def save_dataset(dataset_type, save_relation_dictionary=True):
        dataset = WikiDataVitalsSentences(dataset_type=dataset_type)
        relation_idx_to_name = dataset.relation_idx_to_name

        if save_relation_dictionary:
            with open('wikidatavitals/data/encoded/relation_indices.json', 'w') as f:
                json.dump(relation_idx_to_name, f, indent=4)

        total_sentences = dataset.n_triplets
        loader = DataLoader(dataset, batch_size=64)
        output = np.zeros((total_sentences, 768))
        labels = np.zeros(total_sentences)
        current_output_idx = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                print('{}: batch {}/{}'.format(dataset_type, batch_idx + 1, total_sentences // 64 + 1))
                # computing model output on the sentence batch
                ids, masks = preprocess_sentences_encoder(batch['sentence'], bert_tokenizer)
                batch_size = ids.shape[0]  # the shape of ids and masks is (batch, sentence_length_max=16)
                model_hidden_states = bert(ids, attention_mask=masks).last_hidden_state  # shape (batch, 16, hidden=768)
                model_output = model_hidden_states[:, 0, :]  # use the CLS output: shape (batch, 768)

                # last batch slice handling
                output_upper_slice = min(current_output_idx + batch_size, total_sentences)
                model_upper_slice = batch_size if current_output_idx + batch_size < total_sentences \
                    else total_sentences - current_output_idx

                # saving the output to the final numpy array
                output[current_output_idx:output_upper_slice] = model_output.cpu().numpy()[:model_upper_slice]

                # saving the labels to the final numpy array
                labels[current_output_idx:output_upper_slice] = batch['label'].numpy()[:model_upper_slice]

                current_output_idx += batch_size

                # the actual Dataset length is the total amount of POSSIBLE sentences, so we need to stop short
                if current_output_idx >= total_sentences:
                    break

        np.save('wikidatavitals/data/encoded/' + dataset_type + '.npy', output)
        np.save('wikidatavitals/data/encoded/' + dataset_type + '_labels.npy', labels)

    print('Saving the training set ...')
    save_dataset('train')
    print('Saving the validation set ...')
    save_dataset('val')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--v', '--verbs', action='store_true')
    parser.add_argument('--n-verbs', type=int, default=5, required=False)
    parser.add_argument('--e', '--entities', action='store_true')
    parser.add_argument('--r', '--relations', action='store_true')
    parser.add_argument('--enc', '--encode', action='store_true')
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

    if args.enc:
        print('Encoding Wikidata-vitals sentences using BERT-base ...')
        save_encoded_WDV_sentences()
