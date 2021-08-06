import argparse
import json
import os
import time
from datetime import timedelta
import numpy as np
from tqdm import tqdm
from models.ner import wikifier, CoreferenceResolver
from wikivitals.construction import save_usa_text
import matplotlib.pyplot as plt


def save_usa_entities():
    """
    Saves to 'wikivitals/data/benchmark/usa_entities.json' all the entities present in the USA article.\n
    Requires the USA article text at 'wikivitals/data/benchmark/United States.json'
    from wikivitals.construction.save_usa_text()
    """
    entities = []
    with open('wikivitals/data/benchmark/United States.json', 'r') as f:
        usa_paragraphs = json.load(f)
    coreference_solver = CoreferenceResolver()

    for paragraph in tqdm(usa_paragraphs):
        processed_text = coreference_solver(paragraph)
        wikifier_results = wikifier(processed_text)
        entities.extend([e['id'] for e in wikifier_results])

    if not os.path.exists('wikivitals/data/benchmark/'):
        os.makedirs('wikivitals/data/benchmark/')

    with open('wikivitals/data/benchmark/usa_entities.json', 'w') as f:
        json.dump(entities, f, indent=4)


def save_hundo_entities():
    n_articles = 44859
    np.random.seed(42)
    article_indices = np.random.randint(0, n_articles, 100)

    with open('wikivitals/data/en-articles.txt', 'r') as f:
        article_names = f.readlines()

    article_files = ['wikivitals/data/article_texts/' + article_names[idx] + '.json' for idx in article_indices]

    entities = []
    coreference_solver = CoreferenceResolver()

    for article_file in tqdm(article_files):
        with open(article_file, 'r') as f:
            paragraphs = json.load(f)

        for paragraph in paragraphs:
            processed_text = coreference_solver(paragraph)
            wikifier_results = wikifier(processed_text)
            entities.extend([e['id'] for e in wikifier_results])

    if not os.path.exists('wikivitals/data/benchmark/'):
        os.makedirs('wikivitals/data/benchmark/')

    with open('wikivitals/data/benchmark/hundo_entities.json', 'w') as f:
        json.dump(entities, f, indent=4)

    with open('wikivitals/data/benchmark/hundo_articles.json', 'w') as f:
        json.dump(article_files, f, indent=4)


def save_kg_subset(entity_file, output_file, n_relations=50):
    """
    Saves to 'output_file' all the Wikidata-vitals facts
    that contain entities from the 'entity_file'.\n
    For the USA benchmark, requires:\n
    - 'wikivitals/data/benchmark/usa_entities.json' from benchmark.save_usa_entities()\n
    - 'wikidatavitals/data/relations.json' from wikidatavitals.dataset.save_relations()
    """
    with open('wikidatavitals/data/relation_counts.json', 'r') as f:
        relation_counts = json.load(f)

    relation_ids = [c[0] for c in relation_counts[:n_relations]]
    facts = []

    with open(entity_file, 'r') as f:
        entities = json.load(f)

    with open('wikidatavitals/data/relations.json', 'r') as f:
        relations = json.load(f)

    for (h, r, t) in tqdm(relations):
        if h in entities and t in entities and r in relation_ids:
            facts.append([h, r, t])

    with open(output_file, 'w') as f:
        json.dump(facts, f)  # large and illegible -> no indent


class FactChecker(object):
    def __init__(self, facts_file):
        with open(facts_file, 'r') as f:
            self.facts = json.load(f)

    def check(self, triplet):
        return triplet in self.facts

    def get_rank(self, e1, e2, ordered_candidate_relations):
        for rank, r_id in enumerate(ordered_candidate_relations):
            if [e1, r_id, e2] in self.facts:
                return rank + 1  # ranks start at 1
        return -1  # the true relation isn't in the candidates


class PairChecker(object):
    def __init__(self, facts_file):
        with open(facts_file, 'r') as f:
            facts = json.load(f)

        self.pairs = [(h, t) for (h, _, t) in facts]

    def check(self, e1, e2):
        return (e1, e2) in self.pairs


def usa_benchmark(extractor, config, output_name='usa_benchmark_results'):
    article_text_files = ['wikivitals/data/benchmark/United States.json']
    facts_file = 'wikivitals/data/benchmark/usa_facts.json'
    benchmark_routine(extractor, config, output_name, facts_file, article_text_files)


def hundo_benchmark(extractor, config, output_name='hundo_benchmark_results'):
    with open('wikivitals/data/benchmark/hundo_articles.json', 'r') as f:
        article_text_files = json.load(f)
    facts_file = 'wikivitals/data/benchmark/hundo_facts.json'
    benchmark_routine(extractor, config, output_name, facts_file, article_text_files)


def benchmark_routine(extractor, config, output_name, facts_file, article_text_files):

    def paragraph_routine(_paragraph, _extractor, _all_outputs, _predicted_facts):
        output = extractor.extract_facts(_paragraph, verbose=False)
        _all_outputs.extend(output)
        _predicted_facts.extend([[o['e1_id'], o['property_id'], o['e2_id']] for o in output])

    t0 = time.time()
    print('Starting benchmark ...')
    predicted_facts, all_outputs = [], []
    print('Processing the article(s) ...')

    if len(article_text_files) == 1:  # USA benchmark with just one article
        with open(article_text_files[0], 'r') as f:
            paragraphs = json.load(f)

        for paragraph in tqdm(paragraphs):
            paragraph_routine(paragraph, extractor, all_outputs, predicted_facts)

    else:  # many articles (hundo benchmark)
        for article_file in tqdm(article_text_files):
            with open(article_file, 'r') as f:
                paragraphs = json.load(f)

            for paragraph in paragraphs:
                paragraph_routine(paragraph, extractor, all_outputs, predicted_facts)

    print("Fact checking ...")
    FC = FactChecker(facts_file)
    success_bool = [FC.check(predicted_fact) for predicted_fact in predicted_facts]
    success01 = np.array([int(b) for b in success_bool])

    PC = PairChecker(facts_file)
    pair_success_bool = [PC.check(e1, e2) for (e1, _, e2) in predicted_facts]
    pair_success01 = np.array([int(b) for b in pair_success_bool])

    print("Dumping fact-by-fact results ...")
    for output_idx, output_dict in enumerate(all_outputs):
        output_dict['is_correct'] = str(success_bool[output_idx])

    with open(output_name + '.json', 'w') as f:
        json.dump([config] + all_outputs, f, indent=4)

    n_correct = np.sum(success01)
    n_correct_pairs = np.sum(pair_success01)
    n_predictions = np.shape(success01)[0]

    # Rank-based metrics
    valid_ranks = []
    for output_entry in all_outputs:
        e1, e2, candidates = output_entry['e1_id'], output_entry['e2_id'], output_entry['ordered_candidates']
        rank = FC.get_rank(e1, e2, candidates)
        if rank != -1:
            valid_ranks.append(rank)

    ranks = np.array(valid_ranks).astype(float)

    # Mean Reciprocal Rank
    MRR = np.mean(ranks ** (-1))

    if n_predictions == 0:
        print("No predictions :(")
    else:
        print("Total predictions: {}\t correct%: {:.2f}%".format(n_predictions, 100 * n_correct / n_predictions))
        print("Total pairs: {}\t correct%: {:.2f}%".format(n_predictions, 100 * n_correct_pairs / n_predictions))
        print("Mean Reciprocal Rank (within correct pairs): {:.2f}".format(MRR))

    print('Benchmark time: ', timedelta(seconds=time.time()-t0))

    plt.hist(ranks, 50, density=True, facecolor='b', alpha=0.75)
    plt.xlabel('Rank')
    plt.ylabel('Probability')
    plt.title('Histogram of Ranks')
    plt.show()


def simple_benchmark(extractor, config, output_name='simple_benchmark_results'):
    results = [config]
    total_fact_correct, total_pair_correct = 0, 0
    with open('wikidatavitals/data/simple.json', 'r') as f:
        simple = json.load(f)

    for sentence_entry in simple:
        sent = sentence_entry['sentence']
        detections = extractor.extract_facts(sent, verbose=False)
        pair_correct, fact_correct = False, False
        e1, r, e2 = sentence_entry['e1']['id'], sentence_entry['r']['id'], sentence_entry['e2']['id']

        for det in detections:
            if det['e1_id'] in (e1, e2) and det['e2_id'] in (e1, e2):
                pair_correct = True
            if det['e1_id'] == e1 and det['property_id'] == r and det['e2_id'] == e2:
                fact_correct = True

        if pair_correct:
            total_pair_correct += 1

        if fact_correct:
            total_fact_correct += 1

        results.append({
            'ground_truth': sentence_entry,
            'detections': detections,
            'pair_correct': pair_correct,
            'fact_correct': fact_correct
        })

    print('Correct facts: {.2f}%'.format(100 * total_fact_correct / len(simple)))
    print('Correct pairs: {.2f}%'.format(100 * total_pair_correct / len(simple)))

    with open(output_name + '.json', 'w') as f:
        json.dump(results, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', '--prepare', action='store_true',
                        help='Prepares the USA and Hundo data for benchmarking')
    args = parser.parse_args()

    if args.p:
        print("Saving the USA article's text ...")
        save_usa_text()
        print('Saving the entities from the USA article ...')
        save_usa_entities()
        print('Saving all the facts concerning USA entities ...')
        save_kg_subset('wikivitals/data/benchmark/usa_entities.json', 'wikivitals/data/benchmark/usa_facts.json')
        print('Saving all the entities from the selected 100 wikivitals articles ...')
        save_hundo_entities()
        print('Saving all the wikidata facts from the selected 100 wikivitals articles ...')
        save_kg_subset('wikivitals/data/benchmark/hundo_entities.json', 'wikivitals/data/benchmark/hundo_facts.json')
