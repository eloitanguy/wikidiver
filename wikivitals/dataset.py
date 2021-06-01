import json
import os
import random
from models.ner import CoreferenceResolver, SentenceSplitter, wikifier
from models.comparators import get_sliced_relation_mention
from wikidatavitals.dataset import FactFinder, FactNotFoundError
import numpy as np
import multiprocessing as mp
from multiprocessing.dummy import Pool
import time
from datetime import timedelta
from urllib.error import HTTPError
from torch.utils.data import Dataset


class WikipediaSentences(object):
    """
    A torch (string) Dataset containing sentences from WikiVitals articles.\n
    Each entry is a dictionary:\n
    {\t'sentence': [a sentence excerpt with a triplet mention],
    \t'label': [the idx of the relation]}\n
    The train set uses the first 66% of each article paragraphs and the val set uses the rest.
    """

    def __init__(self, dataset_type, n_sentences_total=200000, n_relations=50, max_entity_pair_distance=4,
                 bilateral_context=4, max_sentence_length=32):
        assert dataset_type in ["train", "val"]
        self.dataset_type = dataset_type
        self.max_entity_pair_distance = max_entity_pair_distance
        self.bilateral_context = bilateral_context
        self.max_sentence_length = max_sentence_length
        self.n_relations = n_relations
        self.n_sentences = int(n_sentences_total * 0.66) if dataset_type == 'train' else int(n_sentences_total * 0.34)

        # Building the relation dictionary
        with open('wikidatavitals/data/relation_counts.json', 'r') as f:
            relation_counts = json.load(f)  # loading the ordered relation counts

        with open('wikidatavitals/data/relation_names.json', 'r') as f:
            relation_names = json.load(f)

        self.relation_ids = [c[0] for c in relation_counts[:self.n_relations]]
        self.relation_idx_to_name = [{'id': ID, 'name': relation_names[ID]} for ID in self.relation_ids]
        self.relation_id_to_idx = {ID: idx for idx, ID in enumerate(self.relation_ids)}

        # Keeping in memory the article titles, we don't include the USA article which is for benchmarks
        with open('wikivitals/data/en-articles.txt', 'r') as f:
            self.article_names = [line for line in f.readlines() if line != 'United States']
        self.n_articles = len(self.article_names)

        # NLP pipelines
        self.coref = CoreferenceResolver()
        self.splitter = SentenceSplitter()
        self.fact_finder = FactFinder()

    def __len__(self):
        return self.n_sentences  # Artificially set to the wanted number of sentences

    def get_random_annotated_sentence(self):
        random.seed()  # seeds the RNG based on system time: parallelization-friendly
        # Choosing an article paragraph at random, applying coreference resolution and sentence splitting
        article_idx = random.randint(0, self.n_articles - 1)
        article_name = self.article_names[article_idx]

        try:  # try loading the chosen text
            with open(os.path.join('wikivitals/data/article_texts/', article_name + '.json'), 'r') as f:
                raw_article_paragraphs = json.load(f)
        except FileNotFoundError:  # in very rare occasions the article download may have failed
            return self.get_random_annotated_sentence()

        random.shuffle(raw_article_paragraphs)

        # The ('sentence excerpt', label) output is obtained from the first Wikidata fact that we find
        for raw_paragraph in raw_article_paragraphs:  # going through the shuffled paragraphs
            paragraph = self.coref(raw_paragraph)
            sentences = self.splitter(paragraph)

            # Selecting sentences based on the train/val split (we split the paragraphs)
            if self.dataset_type == 'train':
                selected_sentences = sentences[:int(0.66 * len(sentences))]
            else:  # val set
                selected_sentences = sentences[int(0.66 * len(sentences)):]
            random.shuffle(sentences)  # shuffling afterwards to ensure that the sets are disjoint

            for sent in selected_sentences:  # going through the sentences
                try:
                    wikifier_results = wikifier(sent)
                except KeyError:  # sometimes the wikifier request will not give the wikidata IDs -> skip the sentence
                    continue

                n_mentions = len(wikifier_results)

                # Trying entity pair possibilities from the sentence
                for e1_idx in range(n_mentions):
                    for e2_idx in range(e1_idx + 1, min(e1_idx + self.max_entity_pair_distance, n_mentions)):
                        e1_dict, e2_dict = wikifier_results[e1_idx], wikifier_results[e2_idx]
                        sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, sent,
                                                                      bilateral_context=self.bilateral_context)

                        if len(sliced_sentence.split(' ')) <= self.max_sentence_length:
                            try:
                                _, r, _ = self.fact_finder.get_fact(e1_dict['id'], e2_dict['id'])
                                if r in self.relation_ids:  # checking if the relation is in the top relations
                                    return {'sentence': sliced_sentence, 'label': self.relation_id_to_idx[r]}
                                pass
                            except FactNotFoundError:  # No fact in this entity pair, carry on
                                pass

        # If we are this far, we have found no fact in the entire article... so we try another
        return self.get_random_annotated_sentence()

    def placeholder_sentence_extractor(self, _):  # for multiprocessing we NEED an argument for the function ...
        return self.get_random_annotated_sentence()


def save_wikipedia_fact_dataset(folder):
    """
    Saves a Wikipedia Sentences dataset with (string) sentences and (int) labels.
    """

    if not os.path.exists(folder):
        os.makedirs(folder)

    def save_dataset(dataset_type, save_relation_dictionary=True):
        dataset = WikipediaSentences(dataset_type=dataset_type)

        relation_idx_to_name = dataset.relation_idx_to_name

        if save_relation_dictionary:
            with open(os.path.join(folder, 'relation_indices.json'), 'w') as f:
                json.dump(relation_idx_to_name, f, indent=4)

        total_sentences = len(dataset)
        workers = mp.cpu_count()
        pool = Pool(workers)
        sentences = []
        labels = np.zeros(total_sentences)
        current_output_idx = 0
        t0 = time.time()

        while current_output_idx <= total_sentences:
            elapsed = time.time() - t0
            ratio = current_output_idx / total_sentences
            print('Extracted sentences: {} [{:.5f}%]\tElapsed: {}\tETA: {}'.format(
                current_output_idx, 100 * ratio, timedelta(seconds=elapsed),
                timedelta(seconds=elapsed / ratio - elapsed) if ratio > 0.000001 else '---')
            )
            try:
                dataset_item_list = pool.map(dataset.placeholder_sentence_extractor, range(workers))
                batch = {
                    'sentence': [item['sentence'] for item in dataset_item_list],
                    'label': np.array([item['label'] for item in dataset_item_list])
                }
                batch_size = len(batch['sentence'])
                upper_slice_exclusive = min(current_output_idx + batch_size, total_sentences)  # avoid going OOB
                labels[current_output_idx:upper_slice_exclusive] = batch['label']
                sentences.extend(batch['sentence'])
                current_output_idx += batch_size

                # checkpointing at every step
                with open(os.path.join(folder, dataset_type + '_sentences.json'), 'w') as f:
                    json.dump(sentences, f)  # will be massive so no indent
                np.save(os.path.join(folder, dataset_type + '_labels.npy'), labels)

            except HTTPError:  # Handle rare exception: webpage retrieval failure
                print('Caught an HTTPError.')
                continue

        pool.close()
        with open(os.path.join(folder, dataset_type + '_sentences.json'), 'w') as f:
            json.dump(sentences, f)  # will be massive so no indent
        np.save(os.path.join(folder, dataset_type + '_labels.npy'), labels)

    print('Saving the training set ...')
    save_dataset('train')
    print('Saving the validation set ...')
    save_dataset('val')


class WikiVitalsAnnotatedSentences(Dataset):
    """
    A torch (string) Dataset containing annotated Wikivitals sentences.\n
    Each entry is a dictionary:\n
    {\t'sentence': [a pseudo-sentence],
    \t'label': [the idx of the relation]}
    """

    def __init__(self, dataset_type):
        assert dataset_type in ["train", "val"]

        with open('wikivitals/data/encoded/{}_sentences.json'.format(dataset_type), 'r') as f:
            self.sentences = json.load(f)  # loading the wikivitals sentences

        # redundancy for compatibility with the general encoding script
        with open('wikivitals/data/encoded/relation_indices.json', 'r') as f:
            self.relation_idx_to_name = json.load(f)

        self.labels = list(np.load('wikivitals/data/encoded/{}_labels.npy'.format(dataset_type)))

        assert len(self.sentences) == len(self.labels), "Got incompatible sentence and label files"

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, item):
        return {
            'sentence': self.sentences[item],
            'label': self.labels[item]
        }
