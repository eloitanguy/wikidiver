import json
from wikivitals.construction import get_article_text_by_name_with_disambiguation
from torch.utils.data import Dataset
import numpy as np
from random import Random
from models.ner import CoreferenceResolver, SentenceSplitter, wikifier
from models.comparators import get_sliced_relation_mention
from wikidatavitals.dataset import FactFinder, FactNotFoundError


class FoundFact(Exception):
    """ A helper exception for breaking out of the loops that search for a wikidata fact in a paragraph"""
    def __init__(self, sentence, relation_id,):
        self.sentence = sentence
        self.relation_id = relation_id


class NoFactInArticleError(Exception):
    """ Raised if the parser found no facts in the current article when calling WikipediaSentences.__getitem__()"""


class WikipediaSentences(Dataset):
    """
    A torch (string) Dataset containing sentences from WikiVitals articles.\n
    Each entry is a dictionary:\n
    {\t'sentence': [a sentence excerpt with a triplet mention],
    \t'label': [the idx of the relation]}\n
    The train set uses the first 66% of each article paragraphs and the val set uses the rest.
    """
    def __init__(self, dataset_type, n_sentences_total=200000, n_relations=50, max_entity_pair_distance=4,
                 bilateral_context=4, max_sentence_length=15):
        assert dataset_type in ["train", "val"]
        self.dataset_type = dataset_type
        np.random.seed(42)
        self.RNG = Random(42)
        self.max_entity_pair_distance = max_entity_pair_distance
        self.bilateral_context = bilateral_context
        self.max_sentence_length = max_sentence_length
        self.n_relations = n_relations
        self.n_sentences = int(n_sentences_total * 0.66) if dataset_type == 'train' else int(n_sentences_total * 0.34)

        # Building the relation dictionary
        with open('wikidatavitals/data/relation_counts.json', 'r') as f:
            relation_counts = json.load(f)  # loading the ordered relation counts

        self.relation_ids = [c[0] for c in relation_counts[:self.n_relations]]
        self.relation_id_to_idx = {ID: idx for idx, ID in enumerate(self.relation_ids)}

        # Keeping in memory the article titles, we don't include the USA article which is for benchmarks
        with open('wikivitals/data/en-articles.txt', 'r') as f:
            self.article_names = [line for line in f.readlines() if line != 'United States']
        self.n_articles = len(self.article_names)

        # NLP pipelines
        self.coref = CoreferenceResolver()
        self.splitter = SentenceSplitter()
        self.fact_finder = FactFinder()

    def __len(self):
        return self.n_sentences  # Artificially set to the wanted number of sentences

    def __getitem__(self, item):
        # Choosing an article paragraph at random, applying coreference resolution and sentence splitting
        article_idx = self.RNG.randint(0, self.n_articles - 1)
        raw_article_paragraphs = get_article_text_by_name_with_disambiguation(self.article_names[article_idx])
        self.RNG.shuffle(raw_article_paragraphs)

        # The ('sentence excerpt', label) output is obtained from the first Wikidata fact that we find
        try:  # Trying to raise a FactFound error which gives us the fact information
            for raw_paragraph in raw_article_paragraphs:  # going through the shuffled paragraphs
                paragraph = self.coref(raw_paragraph)
                sentences = self.splitter(paragraph)
                self.RNG.shuffle(sentences)

                # Selecting sentences based on the train/val split (we split the paragraphs)
                if self.dataset_type == 'train':
                    selected_sentences = sentences[:int(0.66*len(sentences))]
                else:  # val set
                    selected_sentences = sentences[int(0.66*len(sentences)):]

                for sent in selected_sentences:  # going through the sentences
                    wikifier_results = wikifier(sent)
                    n_mentions = len(wikifier_results)

                    # Trying entity pair possibilities from the sentence
                    for e1_idx in range(n_mentions):
                        for e2_idx in range(e1_idx + 1, min(e1_idx + self.max_entity_pair_distance, n_mentions)):
                            e1_dict, e2_dict = wikifier_results[e1_idx], wikifier_results[e2_idx]
                            sliced_sentence = get_sliced_relation_mention(e1_dict, e2_dict, sent,
                                                                          bilateral_context=self.bilateral_context)

                            if len(sliced_sentence.split(' ') <= self.max_sentence_length):
                                try:
                                    _, r, _ = self.fact_finder.get_fact(e1_dict['id'], e2_dict['id'])
                                    if r in self.relation_ids:  # checking if the relation is in the top relations
                                        raise FoundFact(sliced_sentence, r)  # Found a fact! return it
                                    pass
                                except FactNotFoundError:  # No fact in this entity pair, carry on
                                    pass

            raise NoFactInArticleError  # If we are this far, we have found no fact in the entire article...

        except FoundFact as ff:
            return {'sentence': ff.sentence, 'label': self.relation_id_to_idx[ff.relation_id]}
