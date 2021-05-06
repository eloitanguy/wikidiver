from wikidatavitals.ner import wikifier, CoreferenceResolver
from models.comparators import UniversalSentenceEncoderComparator, get_comparison_sentences
import json


def extract_facts(sentence):
    """
    A test version of V1 on one sentence
    """

    # Step 1: NER
    coreference_solver = CoreferenceResolver()
    processed_text = coreference_solver(sentence)
    wikifier_results = wikifier(processed_text)

    # creating entity pairs (all possibilities, in order)
    entity_pairs = []
    n_mentions = len(wikifier_results)
    for e1_idx in range(n_mentions):
        for e2_idx in range(e1_idx+1, n_mentions):
            entity_pairs.append([wikifier_results[e1_idx]['mention'], wikifier_results[e2_idx]['mention']])

    # Step 2: comparison with generated sentences
    with open('wikidatavitals/data/property_verbs.json', 'r') as f:
        verbs = json.load(f)
    comparison_sentences_by_pair = [get_comparison_sentences(e1, e2, verbs) for (e1, e2) in entity_pairs]
    Comparator = UniversalSentenceEncoderComparator()
    print(Comparator.compare_sentences(processed_text, comparison_sentences_by_pair[0]))  # just pair 1 here

    # TODO: similarities to chosen property (argmax + find property index) -> final facts output


extract_facts('Carlos Santana is a Mexican guitarist.')
