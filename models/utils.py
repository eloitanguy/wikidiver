import os
import sys

import numpy as np
from spacy.lang.en import English


def character_idx_to_word_idx(text):
    """
    Example: "Hello there" -> [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1] (we include the following spaces!)
    :param text: text string
    :return: character-per-character list containing the corresponding word indices
    """
    res = []
    words = text.split(' ')
    for word_idx, w in enumerate(words):
        res = res + [word_idx]*(len(w)+1)  # +1 for space
    return res


class SentenceSplitter(object):
    """
    Splits a text into sentences using SpaCy.
    Code from https://github.com/explosion/spaCy/issues/93#issuecomment-138773719
    """
    def __init__(self):
        self.nlp = English()
        self.nlp.add_pipe(self.nlp.create_pipe('sentencizer'))

    def __call__(self, text):
        doc = self.nlp(text)
        return [sent.string.strip() for sent in doc.sents]


def length_of_longest_common_subsequence(s1: str, s2: str):
    """
    Classic Dynamic Programming problem, finding the length of the longest common subsequence between two strings.
    A subsequence of a string is defined as a string with characters contained in the original string in the same order.
    For instance 'H there' is a subsequence of 'Hello there!
    """
    X, Y = list(s1), list(s2)
    # LCS[i][j] is the LCS of [x1 ... xi] and [y1 ... yj] (index starts at 1 to have LCS[0] correspond to empty X)
    LCS = np.zeros((len(s1) + 1, len(s2) + 1), dtype=int)
    for i in range(1, len(X) + 1):
        for j in range(1, len(Y) + 1):
            if X[i - 1] != Y[j - 1]:
                LCS[i][j] = max(LCS[i-1][j], LCS[i][j-1])
            else:
                LCS[i][j] = LCS[i-1][j-1] + 1
    return LCS[-1][-1]


def find_most_similar_word_idx_interval(sent, name):
    """
    Finds the interval of words in the sentence that best matches the 'name'. \n
    The similarity between an interval of words and the name is given by their length of longest common subsequence. \n
    :param sent: a sentence
    :param name: a string referring to a portion of the sentence
    :return: a pair (idx_1, idx2) denoting the (inclusive) indexes of the chosen word interval
    """
    sent_list = sent.split(' ')
    window = len(name.split(' '))
    best_start_idx = -1
    best_LCS = -1

    for word_idx in range(len(sent_list) - window + 1):
        snippet = ' '.join(sent_list[word_idx:word_idx + window])
        LCS = length_of_longest_common_subsequence(snippet, name)

        if LCS > best_LCS:
            best_start_idx = word_idx
            best_LCS = LCS

    return best_start_idx, best_start_idx + window - 1


class HiddenPrints:
    def __enter__(self):
        self._original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout.close()
        sys.stdout = self._original_stdout


def has_intersection(list1, list2):
    return bool(list(set(list1) & set(list2)))


def union_without_repetition(list1, list2):
    return list(set(list1) | set(list2))
