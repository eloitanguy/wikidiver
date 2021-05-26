from torch.utils.data import DataLoader
from transformers import BertTokenizer, BertModel
import torch
import numpy as np
import json
import os


def preprocess_sentences_encoder(sentences, tokenizer, device):
    encoded_dict = tokenizer(sentences, add_special_tokens=True, max_length=32, padding='max_length',
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
