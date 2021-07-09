from models.encoders import save_encoded_sentences, save_pair_dataset
from wikidatavitals.dataset import save_relations, save_verb_idx_to_relation_list, save_property_verbs_dictionary, \
    save_entity_dictionary, WikiDataVitalsSentences, save_decorated_sentence_dataset
import argparse
from wikivitals.construction import save_all_texts
from wikivitals.dataset import save_wikipedia_fact_dataset, WikiVitalsAnnotatedSentences, wikify_sentences
from models.filters import save_entity_types, save_relation_argument_types

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--enc-wd', '--encode-wikidata', action='store_true',
                        help='Stores BERT encodings of wikidata pseudo-sentences')
    parser.add_argument('--v', '--verbs', action='store_true',
                        help='Saves relation verb info: relation-to-verb and verb-index-to-relation dicts')
    parser.add_argument('--e', '--entities', action='store_true',
                        help='Save entity info: id-to-name and id-to-aliases dicts')
    parser.add_argument('--r', '--relations', action='store_true',
                        help='Save relation info: a fact triplet list, an id-to-name dict and an ordered count list')
    parser.add_argument('--sw', '--save-wikivitals', action='store_true',
                        help='Save all wikivitals article texts')
    parser.add_argument('--a-wv', '--annotate-wikivitals', action='store_true',
                        help='Save annotated Wikivitals sentences')
    parser.add_argument('--enc-wv', '--encode-wikivitals', action='store_true',
                        help='Encoding Wikivitals sentences using BERT-base')
    parser.add_argument('--w-wv', '--wikify-wikivitals', action='store_true',
                        help='Run the wikifier on the wikivitals sentences (temporary)')
    parser.add_argument('--enc-p', '--encode-pairs', action='store_true',
                        help='Reads WikiVitals sentences and wikifier results and creates a dataset for pair classif')
    parser.add_argument('--et', '--entity-types', action='store_true',
                        help='Saves all entity types in a json file')
    parser.add_argument('--rt', '--relation-types', action='store_true',
                        help='Saves all relation argument type possibilities in a json file')
    parser.add_argument('--ds', '--decorated-sentences', action='store_true',
                        help='Saves a dataset with sentences decorated with entities and properties')
    args = parser.parse_args()

    if args.enc_wd:
        print('Encoding Wikidata-vitals sentences using BERT-base ...')
        save_encoded_sentences(WikiDataVitalsSentences, 'wikidatavitals/data/encoded/')

    if args.v:
        print('Saving Wikidata verb information ...')
        save_property_verbs_dictionary(5)
        save_verb_idx_to_relation_list()

    if args.e:
        print('Saving Wikidata entity information ...')
        save_entity_dictionary()

    if args.r:
        print('Saving Wikidata relation information ...')
        save_relations()

    if args.sw:
        print('Saving all wikivitals article texts ...')
        save_all_texts()

    if args.a_wv:
        print('Saving annotated Wikivitals sentences ...')
        save_wikipedia_fact_dataset('wikivitals/data/encoded/')

    if args.enc_wv:
        print('Encoding Wikivitals sentences using BERT-base ...')
        save_encoded_sentences(WikiVitalsAnnotatedSentences, 'wikivitals/data/encoded/')

    if args.w_wv:
        print('Running the wikifier on the wikivitals sentences ...')
        folder = 'wikivitals/data/encoded/'
        print('Processing the train sentences ...')
        wikify_sentences(folder + 'train_sentences.json', folder + 'wikified_train_sentences.json')
        print('Processing the val sentences ...')
        wikify_sentences(folder + 'val_sentences.json', folder + 'wikified_val_sentences.json')

    if args.enc_p:
        print('Saving the pair classification dataset ...')
        folder = 'wikivitals/data/encoded/'
        save_pair_dataset(folder + 'wikified_train_sentences.json', folder + 'train_sentences.json')

    if args.et:
        print('Saving all entity types ...')
        save_entity_types()

    if args.rt:
        print('Saving all relation argument type possibilities ...')
        save_relation_argument_types()

    if args.ds:
        print('Saving a dataset with sentences decorated with entities and properties')
        save_decorated_sentence_dataset()
