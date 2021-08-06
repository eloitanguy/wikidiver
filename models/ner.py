from __future__ import unicode_literals, print_function

import spacy
import neuralcoref
import urllib
import json

from models.utils import character_idx_to_word_idx, length_of_longest_common_subsequence
from models.amr import AMRParser

import multiprocessing as mp
from multiprocessing.dummy import Pool


class NoEntity(Exception):
    pass


class CoreferenceResolver(object):
    """
    Class for executing coreference resolution on a given text
    code from https://github.com/huggingface/neuralcoref/issues/288
    """
    def __init__(self):
        # Load SpaCy
        nlp = spacy.load('en')  # if OSError, use CLI: "python -m spacy download en"
        # Add neural coreference to SpaCy's pipe
        neuralcoref.add_to_pipe(nlp)
        self.nlp_pipeline = nlp

    def __call__(self, text):
        doc = self.nlp_pipeline(text)
        # fetches tokens with whitespaces from spacy document
        tok_list = list(token.text_with_ws for token in doc)
        for cluster in doc._.coref_clusters:
            # get tokens from representative cluster name
            cluster_main_words = set(cluster.main.text.split(' '))
            for coref in cluster:
                if coref != cluster.main:  # if coreference element is not the representative element of that cluster
                    if coref.text != cluster.main.text and bool(
                            set(coref.text.split(' ')).intersection(cluster_main_words)) == False:
                        # if coreference element text and representative element text are not equal and none of the
                        # coreference element words are in representative element. This was done to handle nested
                        # coreference scenarios
                        tok_list[coref.start] = cluster.main.text + \
                                                doc[coref.end - 1].whitespace_
                        for i in range(coref.start + 1, coref.end):
                            tok_list[i] = ""

        return "".join(tok_list)


def wikifier(text, threshold=0.9, grouped=True):
    """
    Function that fetches entity linking results from wikifier.com API, includes code from:
    https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e\n
    :return: a list of detection dicts:
        'start_idx': inclusive start index,
        'end_idx':inclusive end index,
        'id': entity Wikidata ID,
        'name': a name for the entity,
        'mention': the sentence slice relating to it
    """

    # Resolve text co-references
    # Prepare the URL.
    data = urllib.parse.urlencode([
        ("text", text), ("lang", 'en'),
        ("userKey", "tgbdmkpmkluegqfbawcwjywieevmza"),
        ("pageRankSqThreshold", "%g" %
         threshold), ("applyPageRankSqThreshold", "true"),
        ("nTopDfValuesToIgnore", "100"), ("nWordsToIgnoreFromList", "100"),
        ("wikiDataClasses", "true"), ("wikiDataClassIds", "false"),
        ("support", "true"), ("ranges", "false"), ("minLinkFrequency", "2"),
        ("includeCosines", "false"), ("maxMentionEntropy", "3")
    ])
    url = "http://www.wikifier.org/annotate-article"

    # Call the Wikifier and read the response.
    req = urllib.request.Request(url, data=data.encode("utf8"), method="POST")
    with urllib.request.urlopen(req, timeout=60) as open_request:
        response = open_request.read()
        response = json.loads(response.decode("utf8"))

    # Output the annotations.
    results = []
    annotated = [False] * len(text.split(' '))  # only annotate once every word index
    char_map = character_idx_to_word_idx(text)

    for annotation in response["annotations"]:
        characters = [(el['chFrom'], el['chTo']) for el in annotation['support']]
        for start_char, end_char in characters:
            start_w, end_w = char_map[start_char], char_map[end_char]
            mention = text[start_char:end_char+1]
            for w_idx in range(start_w, end_w+1):
                if not annotated[w_idx]:  # remove duplicates
                    try:
                        results.append([w_idx, annotation['wikiDataItemId'], annotation['title'], mention])
                        annotated[w_idx] = True
                    except KeyError:  # in rare cases the annotation will not have a wikiDataItemId key, skip it
                        continue

    return get_grouped_ner_results(results) if grouped else results


def get_grouped_ner_results(result_list):
    """
    Used for grouping the outputs of wikifier.
    :param result_list: a list of [idx, wikidatavitals id, name, mention] detections (one per detected word)
    :return: a list of detection dicts:
        'start_idx': inclusive start index,
        'end_idx':inclusive end index,
        'id': entity Wikidata ID,
        'name': a name for the entity,
        'mention': the sentence slice relating to it
    """

    if not result_list:
        return []

    current_start_idx = 0
    previous_id = result_list[0][1]  # assigning a fake -1-th word with the same label as the first one
    res = []

    for word_idx, word_output in enumerate(result_list):
        new_id = word_output[1]
        if new_id != previous_id:  # changed entity
            res.append({'start_idx': current_start_idx,
                        'end_idx': result_list[word_idx-1][0],
                        'id': result_list[word_idx-1][1],
                        'name': result_list[word_idx-1][2],
                        'mention': result_list[word_idx-1][3]})
            current_start_idx = result_list[word_idx][0]
            previous_id = new_id

    # add the last one
    res.append({'start_idx': current_start_idx,
                'end_idx': result_list[-1][0],
                'id': result_list[-1][1],
                'name': result_list[-1][2],
                'mention': result_list[-1][3]})

    return res


class NERParserAMR:
    def __init__(self, threshold=0.8, n_aliases=3):
        self.amr_parser = AMRParser()
        self.threshold = threshold
        self.n_aliases = n_aliases

        with open('wikidatavitals/data/entity_aliases.json', 'r') as f:
            self.entity_aliases = json.load(f)

    @staticmethod
    def _try_alias(snippet, alias):
        return len(snippet) < 3 * len(alias) and len(alias) < 3 * len(snippet)  # none 3x bigger

    def _find_entity(self, snippet):
        best_LCS = -1
        length_of_best = 99999

        for entity_id, aliases in self.entity_aliases.items():
            for alias in aliases[:self.n_aliases]:
                if self._try_alias(snippet, alias):
                    LCS = length_of_longest_common_subsequence(snippet, alias)

                    if LCS > best_LCS or (LCS == best_LCS and len(alias) < length_of_best):
                        best_LCS, best_id, length_of_best = LCS, entity_id, len(alias)
                        if best_LCS / max(length_of_best, len(snippet)) > self.threshold:
                            return best_id

        raise NoEntity

    def _get_var_results(self, var, sentence):
        if var.start_idx == -1 or var.end_idx == -1:  # should never happen but in that case the var is unusable
            return {}
        if var.name:
            name = var.name
        elif var.wiki:
            name = var.wiki
        else:
            name = var.description
            if '-' in name:  # an AMR description with a '-' is a verb and thus unlikely to be an entity -> skip
                return {}

        try:
            entity_id = self._find_entity(name)
        except NoEntity:
            return {}

        return {
            'start_idx': var.start_idx,
            'end_idx': var.end_idx,
            'id': entity_id,
            'name': self.entity_aliases[entity_id][0],  # the first alias is the Wikidata entity name
            'mention': ' '.join(sentence.split(' ')[var.start_idx:var.end_idx + 1])
        }

    def wikify(self, sentence):
        g = self.amr_parser.parse_text(sentence)
        workers = mp.cpu_count()
        pool = Pool(workers)
        results_by_var = pool.starmap(self._get_var_results, [(var, sentence) for var in g.nodes])
        pool.close()
        return [var_res for var_res in results_by_var if var_res != {}]
