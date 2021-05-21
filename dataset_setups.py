from wikidatavitals.dataset import WikiDataVitalsSentences
from models.encoders import save_encoded_sentences
from wikidatavitals.dataset import save_relations, save_verb_idx_to_relation_list, save_property_verbs_dictionary, \
    save_entity_dictionary
import argparse

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
    parser.add_argument('--enc', '--encode', action='store_true')
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
        print('Saving Wikidata relation information')
        save_relations()
