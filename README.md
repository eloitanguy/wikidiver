# WikiDiver: Extracting knowledge from Wikipedia using Wikidata

The general objective is to extract facts of the form of Wikidata IDs (e1, relation, e2) from raw text.

We refer to e1 and e2 as "entities", which can manifest in text in the form of "mentions" that refer to them.
We define an entity's common mentions as "aliases". Similarly, we refer call relation mentions "verbs" (even though
they are not necessarily verbs).

## General setup

#### Getting started

Preliminary note: in order to get spacy and neuralcoref working together, we recommend working in a new conda env with
python 3.7.

First of all it is required to install the python modules with:

    pip install -r requirements.txt

In order to obtain all the dataset files, you can run the included code below or download the files from 
[google drive](https://drive.google.com/drive/folders/1bHteMXBDD0UJ1r-t4aXfWx7Rkj0ag4JY?usp=sharing).

#### Obtaining verbs

We take the aliases of the wikidata-vitals relations from [TorchKGE](https://torchkge.readthedocs.io/en/latest/).

The following command saves a dictionary to ```wikidatavitals/data/property_verbs.json``` that maps a wikidata entity ID to a 
list of verbs that represent it:

    python dataset_setups.py --verbs

The created file weighs 98.7 kB (with a maximum amount of verbs set to 5), the process takes around 7 minutes.

We also save a list at ```wikidatavitals/data/verb_idx2id.json``` mapping the index of a verb to its original property ID.

#### Comparing sentences

We chose the Universal Sentence Encoder as a vector representation for our sentences.
The code for this comparison is located in ```models/comparators.py```

In order to compare two sentences together, we use the cosine similarity of their USE representations.

#### Obtaining entities and their aliases

Thanks to TorchKGE, it is simple to access the id -> title mapping of wikidata-vitals entities.
We save this dictionary in ```wikidatavitals/data/entity_names.json```with:
    
    python dataset_setups.py --entities

The file weighs 1.4 MB.

This command also saves a dictionary mapping entity ids to the corresponding entity's aliases to 
```wikidatavitals/data/entity_aliases.json```. This file weighs 4.0 MB.

Obtaining the aliases requires prompting the Wikidata API, and in total takes around 7 hours.

#### Obtaining relations

Just like for entities, we save a dictionary id -> title for the relations, as well as the list of all the fact triplets
in Wikidata-vitals using:

    python dataset_setups.py --relations

This execution takes about a minute, and the two files weigh 31 kB and 6.9 MB.

## V1: Comparing sentences

We compare a query sentence with a list of generated sentence of the form "entity 1" "verb" "entity 2"

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

The benchmark requires a setup (please also go through the general setup first!):

    python benchmark.py --prepare

It can then be run using:

    python v1.py --benchmark

This process takes 7 minutes with an RTX3090 (python allocates a lot of GPU memory but uses little GPU processing 
power here).

## V2: train a classifier on constructed sentences

Given a fact (e1, r, e2), we build a "sentence" using the entity aliases and relation verbs.
The next step is to compute the BERT [CLS] output on all the built sentences.
These vectors serve as input for a supervised classifier (XGBoost).

In order to test v2 on a sentence "[sentence]", run the command:

    python v2.py --sentence "[sentence]"

#### Build the sentence dataset

In order to save the classification dataset to ```wikidatavitals/data/encoded/```, run:

    python dataset_setups.py --encode

This requires about 5 GB of RAM and 3 GB of VRAM, the process takes under 2min with an RTX 3090. 
The produced files combined weigh 1.2 GB.

#### Train the XGB model

Adjust the configuration in the config file (the default value has good results on the train/val sets), then run:

    python v2.py --train

#### Benchmark V2 with the USA benchmark

    python v2.py --benchmark

## V2.5 Ideas

Instead of training on built sentences, train with distant supervision using Wikipedia text.

## V3 Ideas

- instead of using the BERT [CLS] output, encode every word pair using their attentions
- classify each word pair into relations
- V3.5: use the TransE pair result in the pipeline

## More random ideas

- in the benchmark, cluster the relations and give points for being in the right cluster
- cluster the relations and build a classifier by cluster
- don't try to extract a fact from two entities with there is no Wikipedia link between them
- structured NER
- TransE-aware fact extraction
- graph-NN on the entity/relation tree for hierarchical thinking
