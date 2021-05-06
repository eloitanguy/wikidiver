# WikiDiver: Extracting knowledge from Wikipedia using Wikidata

## V1: Comparing sentences

We compare a query sentence with a list of generated sentence of the form <entity 1> <verb> <entity 2>

#### Obtaining verbs

We take the aliases of the wikidata-vitals relations from [TorchKGE](https://torchkge.readthedocs.io/en/latest/).

The following command saves a dictionary to ```wikidata/data/property_verbs.json``` that maps a wikidata entity ID to a list of verbs that represent it:

    python wikidata/dataset.py --verbs

The created file weighs 98.7 kB (with a maximum amount of verbs set to 5), the process takes around 7 minutes.

#### Comparing sentences

We chose the Universal Sentence Encoder as a vector representation for our sentences.
The code for this comparison is located in ```models/comparators.py```

In order to compare two sentences together, we use the cosine similarity of their USE representations.

#### Obtaining entities

Thanks to TorchKGE, it is simple to access the id -> title mapping of wikidata-vitals entities.
We save this dictionary in ```wikidata/data/entities.json``` quickly with:
    
    python wikidata/dataset.py --entities

The file weighs 1,4 MB and the execution takes a few seconds.

#### Extracting facts

*WIP*

## V2 Ideas

- transform Wikidata into an annotated knowledge triplet dataset using Wikipedia sentence (Distant supervision). For this we need an entity recognition method.
- compute BERT [CLS] outputs on all the annotated sentences
- train a model on the relation classification task

## V3 Ideas

- instead of using the BERT [CLS] output, encode every word pair using their attentions
- classify each word pair into relations