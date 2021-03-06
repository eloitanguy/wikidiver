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

In order to obtain all the dataset and model files, you can run the included code below or download the files from 
[google drive](https://drive.google.com/drive/folders/1bHteMXBDD0UJ1r-t4aXfWx7Rkj0ag4JY?usp=sharing), which is a
download of all our models and data.

This repo uses spacy's 'en' model, please download it using:

    python -m spacy download en

We also use the SPRING AMR parsing in our pipeline 
(see the [SPRING article](https://github.com/SapienzaNLP/spring/blob/main/docs/preprint.pdf)) using scripts adapted 
directly from [the SPRING github](https://github.com/SapienzaNLP/spring). In order to make the AMR parsing scripts
function, please download the [AMR parsing weights](http://nlp.uniroma1.it/AMR/AMR3.parsing-1.0.tar.bz2) (link from
the SPRING github), and place the weights into ```models/spring_amr/AMR3.pt```

### Dataset Setups

Note that all the data setups are done using ```dataset_setups.py```, which details what each script does. You can set 
up this repo by reading the helpers from that file or by reading this README.

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

#### Obtaining entity and relation types

In order to filter triplet possibilities, we need to store the types of each entity and the possible types of the 
arguments of each relation. This is done with:

    python dataset_setups.py --entity-types
    python dataset_setups.py --relation-types

#### Evaluating versions

In order to assess the quality of the knowledge extraction, we put it to the test on the USA article: we consider a
predicted fact correct if the fact is already present in Wikidata-vitals.

The benchmark requires a setup (please also go through the general setup first!):

    python benchmark.py --prepare

It can then be run using (example for v1):

    python v1.py --usa-benchmark

This process takes 7 minutes with an RTX3090 (python allocates a lot of GPU memory but uses little GPU processing 
power here).

Another similar benchmark is the "Hundo" benchmark which does the same as the USA benchmark but with 100 random
articles (note that these articles are randomly taken, and thus can be anywhere in a train/val split).

## V0: Giving the most popular legal relations

This is a baseline that outputs the most likely relation that satisfies type constraints. A triplet is considered legal
if the entities have legal types for the relation. An entity type is defined by an "instance-of" relation in Wikidata,
and the legal types for the head (h) and tail (t) entities for relation (r) are all the types of the corresponding entities
in triplets (h', r, t') present in Wikidata.

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

## V2.5

This model is trained using Wikipedia sentences annotated using known Wikidata facts.
It is a slight variation on the v2 idea, however preparing its dataset is extremely costly.

#### Saving the article texts

    python datasets_setups.py --save-wikivitals

#### Preparing an annotated text dataset

    python dataset_setups.py --annotate-wikivitals

#### Encode the annotated text dataset

    python dataset_setups.py --encode-wikivitals

#### Train v2.5

    python v2.py --train --point-five

## V3

This model classifies entity pairs (i.e., a pair of word groups detected by wikifier) into relations. The word pairs
are encoded using BERT's attentions, and we classify these pair encodings using an XGB model.
- V3.5 idea: use the TransE pair result in the pipeline

#### Encode the pair dataset

The following command takes the sentences from ```wikivitals/data/encoded/train_sentences.json``` and creates an
annotated pair dataset from it:

    python dataset_setups.py --encode-pairs

#### Train V3

    python v3.py --train

## USA Benchmark summary

These results hold true for the default configuration in ```config.py```:

#### No filter

| Version | Detections | Correct % | Correct pairs % |
|:-------:|:----------:|:---------:|:---------------:|
|    V0   |     459    |   0.65%   |      20.7%      |
|    V1   |     51     |     0%    |      7.84%      |
|    V2   |     87     |   3.45%   |      13.79%     |
|   V2.5  |     161    |   22.36%  |      25.47%     |
|    V3   |     698    |   3.72%   |      8.13%      |

#### With Type Filtering

| Version | Detections | Correct % | Correct pairs % |  MRR  |
|:-------:|:----------:|:---------:|:---------------:|:-----:|
|    V0   |     459    |   0.65%   |      20.7%      | 0.296 |
|    V1   |     51     |     0%    |      6.38%      | ---   |
|    V2   |     81     |   3.7%    |      9.88%      | 0.477 |
|   V2.5  |     156    |   23.72%  |      26.28%     | 0.929 |
|    V3   |     615    |   4.35%   |      7.42%      | 0.657 |

## Hundo Benchmark summary

These results hold true for optimised thresholds (not the default configuration in ```config.py```):

#### With Type Filtering

| Version | Detections | Correct % | Correct pairs % |  MRR  |
|:-------:|:----------:|:---------:|:---------------:|:-----:|
|    V0   |    4826    |   0.35%   |      13.8%      |  0.32 |
|    V1   |     488    |   0.28%    |      1.40%     |  0.75 |
|    V2   |     1755   |   0.46%   |      1.82%      |  0.41 |
|   V2.5  |     2179   |   2.52%   |      3.99%      |  0.73 |
|    V3   |     12621  |   0.61%   |      2.12%      |  0.42 |

## AMR models

In order to parse a sentence into AMR format, use the ```AMRParser``` from ```models/amr.py```. This parser converts
a raw text sentence into AMR format using [SPRING](https://github.com/SapienzaNLP/spring), and also detects
Wikidata-vitals entities.

Using the ```amr_relation_detection_helper.py```, you can build quickly a simple model that detects a specific relation,
by giving it sentences that portray this relation.