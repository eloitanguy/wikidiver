import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # Dirty but prevents the tensorflow info spam
import tensorflow_hub as hub
import numpy as np
from typing import List
from numpy.linalg import norm


def cosine_similarity(query, candidates):
    p = np.matmul(candidates, query)
    n = norm(query) * norm(candidates, axis=1)  # vector of ||query|| * ||candidate i||
    res = p / n
    res[np.isnan(res)] = -1.
    return res


class UniversalSentenceEncoderComparator(object):
    def __init__(self):
        module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
        self.model = hub.load(module_url)

    def compare_sentences(self, query: str, candidates: List[str]):
        """
        Gives the cosine similarities between the query and the candidates
        :param query: sentence
        :param candidates: list of n sentences
        :return: the n cosine similarities
        """
        out = self.model([query] + candidates)
        v_query, v_candidates = out[0, :].numpy(), out[1:, :].numpy()
        return cosine_similarity(v_query, v_candidates)


def get_comparison_sentences(e1, e2, property_verbs, double_check=True):
    res = []
    for verbs in property_verbs.values():
        res.extend([e1 + ' ' + verb + ' ' + e2 for verb in verbs])
    if double_check:
        res.append(' '.join([e1, e2]))
    return res


def get_sliced_relation_mention(e1_dict, e2_dict, sentence, bilateral_context=0):
    """
    :param e1_dict: {start_idx, end_idx, wikidatavitals id, name, mention}
    :param e2_dict: {start_idx, end_idx, wikidatavitals id, name, mention}
    :param sentence: text sentence mentioning e1 and e2
    :param bilateral_context: number of words before e1 and after e2 to include in the slice
    :return: the same text sliced between e1 and e2 (inclusive)
    """
    start_word_idx = max(e1_dict['start_idx'] - bilateral_context, 0)
    text_list = sentence.split(' ')
    text_length = len(text_list)
    # the index is inclusive and Python is exclusive, hence the +1
    end_word_idx = min(e2_dict['end_idx'] + 1 + bilateral_context, text_length - 1)
    res_list = text_list[start_word_idx:end_word_idx]
    return ' '.join(res_list)
