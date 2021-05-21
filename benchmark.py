import json
from models.ner import wikifier, CoreferenceResolver
import os
import numpy as np
import argparse
from tqdm import tqdm
from wikivitals.construction import save_usa_text
import time
from datetime import timedelta


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


def save_kg_subset():
    """
    Saves to 'wikivitals/data/benchmark/usa_facts.json' all the Wikidata-vitals facts
    that contain entities from the USA article.\n
    Requires:\n
    - 'wikivitals/data/benchmark/usa_entities.json' from benchmark.save_usa_entities()\n
    - 'wikidatavitals/data/relations.json' from wikidatavitals.dataset.save_relations()
    """
    usa_facts = []

    with open('wikivitals/data/benchmark/usa_entities.json', 'r') as f:
        usa_entities = json.load(f)

    with open('wikidatavitals/data/relations.json', 'r') as f:
        relations = json.load(f)

    for triplet in tqdm(relations):
        if triplet[0] in usa_entities and triplet[2] in usa_entities:
            usa_facts.append(triplet)

    with open('wikivitals/data/benchmark/usa_facts.json', 'w') as f:
        json.dump(usa_facts, f)  # large and illegible -> no indent


class FactChecker(object):
    def __init__(self):
        with open('wikivitals/data/benchmark/usa_facts.json', 'r') as f:
            self.facts = json.load(f)

    def check(self, triplet):
        return triplet in self.facts


def usa_benchmark(extractor, config, output_name='usa_benchmark_results'):
    t0 = time.time()
    print('Starting USA benchmark ...')
    with open('wikivitals/data/benchmark/United States.json', 'r') as f:
        usa_paragraphs = json.load(f)
    predicted_facts_usa = []

    print('Processing the USA article ...')
    all_outputs = []
    for paragraph in tqdm(usa_paragraphs):
        output = extractor.extract_facts(paragraph, verbose=False)
        all_outputs.extend(output)
        predicted_facts_usa.extend([[o['e1_id'], o['property_id'], o['e2_id']] for o in output])

    print("Fact checking ...")
    FC = FactChecker()
    success_bool = [FC.check(predicted_fact) for predicted_fact in predicted_facts_usa]
    success01 = np.array([int(b) for b in success_bool])

    print("Dumping fact-by-fact results ...")
    for output_idx, output_dict in enumerate(all_outputs):
        output_dict['is_correct'] = str(success_bool[output_idx])

    with open(output_name + '.json', 'w') as f:
        json.dump([config] + all_outputs, f, indent=4)

    n_correct = np.sum(success01)
    n_predictions = np.shape(success01)[0]
    if n_predictions == 0:
        print("No predictions :(")
    else:
        print("Total predictions: {}\t correct%: {:.2f}%".format(n_predictions, 100 * n_correct/n_predictions))

    print('Benchmark time: ', timedelta(seconds=time.time()-t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--p', '--prepare', action='store_true', help='Prepares the data for benchmarking')
    args = parser.parse_args()

    if args.p:
        print("Saving the USA article's text ...")
        save_usa_text()
        print('Saving the entities from the USA article ...')
        save_usa_entities()
        print('Saving all the facts concerning USA entities ...')
        save_kg_subset()
