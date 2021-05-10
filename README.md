# WikiDiver: Extracting knowledge from Wikipedia using Wikidata

## V1: Comparing sentences

We compare a query sentence with a list of generated sentence of the form "entity 1" "verb" "entity 2"

#### Obtaining verbs

We take the aliases of the wikidata-vitals relations from [TorchKGE](https://torchkge.readthedocs.io/en/latest/).

The following command saves a dictionary to ```wikidata/data/property_verbs.json``` that maps a wikidata entity ID to a 
list of verbs that represent it:

    python wikidata/dataset.py --verbs

The created file weighs 98.7 kB (with a maximum amount of verbs set to 5), the process takes around 7 minutes.

We also save a list at ```wikidata/data/verb_idx2id.json``` mapping the index of a verb to its original property ID.

#### Comparing sentences

We chose the Universal Sentence Encoder as a vector representation for our sentences.
The code for this comparison is located in ```models/comparators.py```

In order to compare two sentences together, we use the cosine similarity of their USE representations.

#### Obtaining entities

Thanks to TorchKGE, it is simple to access the id -> title mapping of wikidata-vitals entities.
We save this dictionary in ```wikidata/data/entity_names.json``` quickly with:
    
    python wikidata/dataset.py --entities

The file weighs 1.4 MB and the execution takes a few seconds.

#### Obtaining relations (benchmarking only, for now)

Just like for entities, we save a dictionary id -> title for the relations, as well as the list of all the fact triplets
in Wikidata-vitals using:

  python wikidata/dataset.py --relations

This execution takes about a minute, and the two files weigh 31 kB and 6.9 MB.

#### Extracting facts

We apply the comparison method to all the (ordered) entities in the original sentence, then for each pair 
(that isn't too far away in the text) we find the most similar property.

We added a threshold method to avoid creating facts, eg 'Carlos Santana is a Mexican guitarist.': 
we don't want a property between "Mexican (nationality)" and "guitarist (occupation)", so far this method isn't 
too effective.

In order to test v1 on a sentence "[sentence]", run the command:

    python v1.py --sentence "[sentence]"

/!\ This uses a TensorFlow model (the Universal Sentence Encoder), so having a GPU available is recommended.

#### Evaluate v1

In order to assess the quality of the knowledge extraction, we put it to the test on the USA article: we consider a
predicted fact correct if the fact is already present in Wikidata-vitals.

The benchmark requires a setup (please also go through the relation setup first!):

    python benchmark.py --prepare

It can then be run using:

    python v1.py --benchmark

This process takes 7 minutes with an RTX3090 (python allocates a lot of GPU memory but uses little GPU processing 
power here).

## V2 Ideas

- transform Wikidata into an annotated knowledge triplet dataset using Wikipedia sentence (Distant supervision). 
  For this we need an entity recognition method.
- compute BERT [CLS] outputs on all the annotated sentences
- train a model on the relation classification task

## V3 Ideas

- instead of using the BERT [CLS] output, encode every word pair using their attentions
- classify each word pair into relations
