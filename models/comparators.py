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
        print([query] + candidates)
        out = self.model([query] + candidates)
        v_query, v_candidates = out[0, :].numpy(), out[1:, :].numpy()
        return cosine_similarity(v_query, v_candidates)


def get_comparison_sentences(e1, e2, property_verbs):
    res = []
    for verbs in property_verbs.values():
        res.extend([e1 + ' ' + verb + ' ' + e2 for verb in verbs])
    return res
