# Extracting knowledge from Wikipedia

Plan so far:

- transform Wikidata into an annotated knowledge triplet dataset using Wikipedia sentence (Distant supervision). For this we need an entity recognition method.
- compute BERT [CLS] outputs on all of the annotated sentences
- train a model on the relation classification task