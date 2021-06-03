from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json
import os
from tqdm import tqdm
from wikidatavitals.dataset import FactFinder, FactNotFoundError


def preprocess_sentences_encoder(sentences, tokenizer, device, max_length=32):
    encoded_dict = tokenizer(sentences, add_special_tokens=True, max_length=max_length, padding='max_length',
                             truncation=True, return_attention_mask=True, return_tensors='pt')

    return encoded_dict['input_ids'].to(device), encoded_dict['attention_mask'].to(device)


def handle_batch(batch, bert, bert_tokenizer, device, current_output_idx, total_sentences, output, labels):
    # computing model output on the sentence batch
    ids, masks = preprocess_sentences_encoder(batch['sentence'], bert_tokenizer, device)
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

    return output, labels, current_output_idx


def save_encoded_sentences(torch_dataset, folder):
    """
    Feeds the sentences from the given Dataset to BERT-base and saves a numpy .npy file for the train and val sets.
    Uses pytorch datasets and data loaders.
    """

    device = torch.device('cuda:0')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained("bert-base-uncased").cuda().eval()

    if not os.path.exists(folder):
        os.makedirs(folder)

    def save_dataset(dataset_type, save_relation_dictionary=True):
        dataset = torch_dataset(dataset_type=dataset_type)
        relation_idx_to_name = dataset.relation_idx_to_name

        if save_relation_dictionary:
            with open(os.path.join(folder, 'relation_indices.json'), 'w') as f:
                json.dump(relation_idx_to_name, f, indent=4)

        total_sentences = len(dataset)
        loader = DataLoader(dataset, batch_size=32, num_workers=32)
        n_batches = len(loader)
        output = np.zeros((total_sentences, 768))
        labels = np.zeros(total_sentences)
        current_output_idx = 0

        with torch.no_grad():
            for batch_idx, batch in enumerate(loader):
                print('Batch {}/{}'.format(batch_idx, n_batches))
                output, labels, current_output_idx = handle_batch(batch, bert, bert_tokenizer, device,
                                                                  current_output_idx, total_sentences, output, labels)

        np.save(os.path.join(folder, dataset_type + '.npy'), output)
        np.save(os.path.join(folder, dataset_type + '_labels.npy'), labels)

    print('Saving the training set ...')
    save_dataset('train')
    print('Saving the validation set ...')
    save_dataset('val')


class PairEncoder(object):
    """
    Uses BERT attentions in order to encode an ordered pair of words from a sentence. For a given entity index pair, the
    output is a 1D tensor of shape 2 pair orders * 12 layers * 12 heads = 288.
    If a given entity spans across several words, we take the averages.
    """

    def __init__(self, max_sentence_length=16):
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained("bert-base-uncased").cuda().eval()
        self.max_sentence_length = max_sentence_length
        self.device = torch.device('cuda:0')

    def get_pair_encoding(self, sentence, e1_slice, e2_slice):
        """
        :param sentence: a list of words
        :param e1_slice: the slice (tuple) of the first entity in the pair
        :param e2_slice: the slice (tuple) of the second entity in the pair
        :return: a 1D torch tensor of shape (144) with the attentions word 1 -> word 2
        """
        with torch.no_grad():
            ids, masks = preprocess_sentences_encoder(sentence, self.tokenizer, self.device,
                                                      max_length=self.max_sentence_length)
            e1_lower, e1_upper = e1_slice
            e2_lower, e2_upper = e2_slice
            output = self.bert(ids, attention_mask=masks, output_attentions=True)
            return torch.stack([a[0, :, e1_lower:e1_upper, e2_lower:e2_upper].mean() for a in output.attentions] +
                               [a[0, :, e2_lower:e2_upper, e1_lower:e1_upper].mean() for a in output.attentions])


class TripletFinder(object):
    def __init__(self, n_relations=50):
        self.FF = FactFinder(require_top=n_relations)

    def get_triplets(self, wikifier_result):
        """
        :param wikifier_result: the output (dictionary list) of the wikifier function
        :return: a list of found facts: a list of dicts {'r_id': Wikidata relation ID,
        'e1_slice': slice of e1's mention in the sentence, 'e2_slice': slice of e2's mention in the sentence}
        """
        # going through all entity detections
        res = []
        for d1_idx, d1 in enumerate(wikifier_result):
            for d2_idx, d2 in enumerate(wikifier_result):
                if d1_idx == d2_idx:
                    continue
                e1_id, e2_id = d1['id'], d2['id']
                try:
                    _, r, _ = self.FF.get_fact(e1_id, e2_id)
                    res.append({'r_id': r,
                                'e1_slice': (d1['start_idx'], d1['end_idx'] + 1),
                                'e2_slice': (d2['start_idx'], d2['end_idx'] + 1)})
                except FactNotFoundError:
                    continue
        return res


def save_pair_dataset(wikifier_results_file, sentence_file, n_relations=50):
    """
    :param wikifier_results_file: The output json file of wikify_sentences containing entity detections
    :param sentence_file: The output json file of save_wikipedia_fact_dataset
    :param n_relations: the top _ relations to limit the study to
    :return: Saves pair encodings and labels in numpy files (having split the data into train/val)
    """
    # loading the data to process
    with open(wikifier_results_file, 'r') as f:
        wiki_res = json.load(f)

    with open(sentence_file, 'r') as f:
        sentences = json.load(f)

    # Building the relation dictionary
    with open('wikidatavitals/data/relation_counts.json', 'r') as f:
        relation_counts = json.load(f)  # loading the ordered relation counts

    relation_ids = [c[0] for c in relation_counts[:n_relations]]
    relation_id_to_idx = {ID: idx for idx, ID in enumerate(relation_ids)}

    TF = TripletFinder()
    PE = PairEncoder()

    pair_encodings = []  # will be a list of numpy arrays of shape (288), stacked into one at the end
    pair_labels = []  # will be a list of relation indices (ints), stacked into one array at the end

    for idx, result_dicts in enumerate(tqdm(wiki_res)):
        sent = sentences[idx].split(' ')
        found_triplets = TF.get_triplets(result_dicts)
        if not found_triplets:
            continue
        for found_triplet in found_triplets:
            r = relation_id_to_idx[found_triplet['r_id']]
            pair_labels.append(r)
            pair_encoding = PE.get_pair_encoding(sent, found_triplet['e1_slice'], found_triplet['e2_slice'])
            pair_encodings.append(pair_encoding.cpu().numpy())

    n_total = len(pair_encodings)
    train_val_split = int(n_total * 0.66)
    train_pair_encodings, val_pair_encodings = pair_encodings[train_val_split:], pair_encodings[:train_val_split]
    train_pair_labels, val_pair_labels = pair_labels[train_val_split:], pair_labels[:train_val_split]
    np.save('wikivitals/data/encoded/train_pair_encodings.npy', train_pair_encodings)
    np.save('wikivitals/data/encoded/val_pair_encodings.npy', val_pair_encodings)
    np.save('wikivitals/data/encoded/train_pair_labels.npy', train_pair_labels)
    np.save('wikivitals/data/encoded/val_pair_labels.npy', val_pair_labels)
