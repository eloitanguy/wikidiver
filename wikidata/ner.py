import spacy
import neuralcoref
import urllib
import json


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


def wikifier(text, threshold=0.9):
    """
    Function that fetches entity linking results from wikifier.com API, includes code from:
    https://towardsdatascience.com/from-text-to-knowledge-the-information-extraction-pipeline-b65e7e30273e
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
            for w_idx in range(start_w, end_w+1):
                if not annotated[w_idx]:  # remove duplicates
                    results.append([w_idx, annotation['wikiDataItemId'], annotation['title']])
                    annotated[w_idx] = True
    return results


if __name__ == '__main__':
    t = 'A member of the Democratic Party, Barack Obama was the first African-American president of the United States'
    print('Example sentence: ', t)
    coreference_resolver = CoreferenceResolver()
    processed_text = coreference_resolver(t)

    with open('test_output.json', 'w') as f:
        json.dump(wikifier(processed_text), f, indent=4)
