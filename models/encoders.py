from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json
import os


def preprocess_sentences_encoder(sentences, tokenizer, device):
    encoded_dict = tokenizer(sentences, add_special_tokens=True, max_length=16, padding='max_length',
                             truncation=True, return_attention_mask=True,
                             return_tensors='pt')

    return encoded_dict['input_ids'].to(device), encoded_dict['attention_mask'].to(device)


def save_encoded_sentences(torch_dataset):
    """
    Feeds the sentences from the given Dataset to BERT-base and saves a numpy .npy file for the train and val sets
    """

    device = torch.device('cuda:0')
    bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    bert = BertModel.from_pretrained("bert-base-uncased").cuda().eval()

    if not os.path.exists('wikidatavitals/data/encoded/'):
        os.makedirs('wikidatavitals/data/encoded/')

    def save_dataset(dataset_type, save_relation_dictionary=True):
        dataset = torch_dataset(dataset_type=dataset_type)
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

                # the actual Dataset length is the total amount of POSSIBLE sentences, so we need to stop short
                if current_output_idx >= total_sentences:
                    break

        np.save('wikidatavitals/data/encoded/' + dataset_type + '.npy', output)
        np.save('wikidatavitals/data/encoded/' + dataset_type + '_labels.npy', labels)

    print('Saving the training set ...')
    save_dataset('train')
    print('Saving the validation set ...')
    save_dataset('val')
