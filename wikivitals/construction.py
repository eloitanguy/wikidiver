# Adapted code by Armand Boschin

import pandas as pd
import numpy as np
import requests
import wikipedia
import os

from bs4 import BeautifulSoup
from tqdm.notebook import tqdm

from scipy import sparse
from collections import defaultdict

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

wikipedia.set_lang("en")
wiki_url = 'https://en.wikipedia.org'
sep = '|||'


def get_categories(_url):
    _page = requests.get(_url)
    _soup = BeautifulSoup(_page.text, 'html.parser')
    _categories = dict()
    all_ = _soup.find_all('td', attrs={'align': 'left'})
    for _i in all_:
        try:
            _categories[_i.find('a').text] = _i.find('a').attrs['href']
        except AttributeError:
            pass
    return _categories


def remove_span(word):
    if 'span>' in word:
        return word.split('span>')[-1]
    elif '}}' in word:
        return word.split('}}')[-1]
    else:
        return word


def remove_specials(word, car='=<>*#'):
    for _c in car:
        word = word.replace(_c, '')
    return word


def remove_par(word):
    return word.split('(')[0][:-1]


def clean_text(word):
    if word[0] == '=':
        word = remove_par(remove_specials(remove_span(word))).strip()
    else:
        if '[[' in word:
            word = word.split('[[')[1].split(']]')[0]
        else:
            word = remove_specials(word).strip()
    return word.split('|')[0]


def get_article(_filename, general_category):
    category = []
    with open(_filename) as _f:
        for _row in _f:
            if len(_row):
                if _row[0] == '=':
                    # new category
                    k = 0
                    while _row[k] == '=':
                        k += 1
                    if k > 1:
                        category = category[:k - 1]
                    category += [clean_text(_row)]
                    sub_category = []
                elif _row[0] == '#':
                    # new entry
                    articles.append(clean_text(_row))
                    k = 0
                    while _row[k] == '#':
                        k += 1
                    sub_category = sub_category[:k - 1] + [clean_text(_row)]
                    if category[0] == general_category:
                        categories.append(sep.join(category + sub_category[:-1]))
                    else:
                        categories.append(sep.join([general_category] + category + sub_category[:-1]))


# ################# #
# English Wikipedia #
# ################# #

categories = get_categories('https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5')
general = {k: v.split('/')[5] for k, v in categories.items()}
filenames = list(categories.keys())

if not os.path.exists('data/mds/'):
    os.makedirs('data/mds/')

for k, v in categories.items():  # saves the category pages' text
    with open('data/mds/{}'.format(k), 'w', encoding='utf8') as f:
        url = "https://en.wikipedia.org/w/index.php?title={}&action=edit".format(v[6:])
        page = requests.get(url)
        soup = BeautifulSoup(page.text, 'html.parser')
        f.write(soup.find('textarea').text)

articles = []
categories = []

for filename in filenames:
    get_article('data/mds/' + filename, general[filename])

with open('data/en-categories.txt', 'w', encoding='utf8') as f:
    for cat in categories:
        f.write(cat + "\n")

with open('data/en-articles.txt', 'w', encoding='utf8') as f:
    for name in articles:
        f.write(name + "\n")


def scrapping(_names, _filename, size=1000, _sep='|||'):
    text_file = open('data/' + _filename + '-texts.txt', 'w', encoding='utf8')
    link_file = open('data/' + _filename + '-links.txt', 'w', encoding='utf8')

    _texts = {}
    _links = {}
    for idx, name1 in tqdm(enumerate(_names), total=len(_names)):
        try:
            article = wikipedia.page(name1)
            _texts[name1] = article.summary
            _links[name1] = article.links
        except:
            pass

        if not idx % size:
            for name2 in _texts:
                text_file.write(name2 + _sep + _texts[name2].replace('\n', ' ') + "\n")
            for name2 in _links:
                link_file.write(name2 + _sep + _sep.join(_links[name2]) + "\n")
            _texts = {}
            _links = {}

    for name1 in _texts:
        text_file.write(name1 + _sep + _texts[name1].replace('\n', ' ') + "\n")
    for name1 in _links:
        link_file.write(name1 + _sep + _sep.join(_links[name1]) + "\n")


scrapping(articles, 'en')

# ###### #
# Export #
# ###### #

texts = {}
with open('data/en-texts.txt') as f:
    for row in f:
        words = row.split(sep)
        texts[words[0]] = words[1][:-1]

links = {}
with open('data/en-links.txt') as f:
    for row in f:
        words = row.split(sep)
        links[words[0]] = words[1:][:-1]

names = []
with open('data/en-articles.txt') as f:
    for row in f:
        names.append(row[:-1])

categories = []
with open('data/en-categories.txt') as f:
    for row in f:
        categories.append(row[:-1])

ix2name = dict()
ix2cat = dict()
name2ix = dict()

for i in range(len(names)):
    ix2name[i] = names[i]
    ix2cat[i] = categories[i].replace('_', ' ')
    name2ix[names[i]] = i

#  Only keep the names/pages that have an entry in links.
tmp = list(links.keys())
tmp.sort()
ix2cat = {i: ix2cat[name2ix[tmp[i]]] for i in range(len(tmp))}
ix2name = {i: n for i, n in enumerate(tmp)}
name2ix = {v: k for k, v in ix2name.items()}
row = []
col = []
for name in tmp:
    try:
        for L in links[name]:
            name_ = L.split('|')[0]
            if name_ in name2ix.keys():
                row.append(name2ix[name])
                col.append(name2ix[name_])
    except KeyError:
        pass

# ######### #
# Adjacency #
# ######### #

df = pd.DataFrame(np.array([row, col])).T
df.columns = ['from', 'to']

df = df.sort_values(['from', 'to'])
df.reset_index(drop=True, inplace=True)

df['value'] = True
df.to_csv('data/output/adjacency.csv', sep=',', header=False, index=False)

# ###### #
# Labels #
# ###### #

cats = dict()
for c in general.values():
    if c.replace('_', ' ') not in cats.keys():
        cats[c.replace('_', ' ')] = len(cats)

labels = pd.DataFrame([cats[c] for c in [ix2cat[i].split(sep)[0] for i in range(len(ix2cat))]])
labels.to_csv('data/output/labels.csv', sep=',', header=False, index=False)
pd.DataFrame(list(cats.keys())).to_csv('data/output/names_labels.csv', sep=',', header=False, index=False)

# ################ #
# Labels Hierarchy #
# ################ #

hierarchies = list(set(['.'.join(ix2cat[i].split(sep)) for i in range(len(ix2cat))]))

tmp_ = defaultdict(list)
for h in hierarchies:
    tmp_[cats[h.split('.')[0]]].append(h)

for k, v in tmp_.items():
    v.sort()

cats = []
for i in range(len(tmp_)):
    cats.extend(tmp_[i])

assert len(cats) == len(hierarchies)

cats = {c: i for i, c in enumerate(cats)}
labels = pd.DataFrame([cats['.'.join(c.split(sep))] for c in [ix2cat[i] for i in range(len(ix2cat))]])
labels.to_csv('data/output/labels_hierarchy.csv', sep=',', header=False, index=False)
pd.DataFrame(list(cats.keys())).to_csv('data/output/names_labels_hierarchy.csv', sep=',', header=False, index=False)
pd.DataFrame(tmp).to_csv('data/output/names.csv', sep=',', header=False, index=False)

# ############### #
# Post-processing #
# ############### #

text = {}
with open('data/en-texts.txt', 'r', encoding='utf8') as f:
    for row in f:
        name, summary = tuple(row.split(sep))
        text[name] = summary[:-1]

nltk.download('stopwords')
language = 'english'
stop_words = stopwords.words(language)

porter = PorterStemmer()
text_stems = {}
for name in text:
    tokens = [word.lower() for word in word_tokenize(text[name])]
    text_stems[name] = [porter.stem(token) for token in tokens if token.isalpha() and token not in stop_words]

stems = []
for s in text_stems.values():
    stems += s

unique_stems = set(stems)
stem_index = {stem: i for i, stem in enumerate(unique_stems)}

edges = [(name2ix[n], stem_index[s]) for n, l in text_stems.items() for s in l]
row = [edge[0] for edge in edges]
col = [edge[1] for edge in edges]
biadjacency = sparse.csr_matrix((np.ones(len(edges)), (row, col)))
counts = biadjacency.T.dot(np.ones(biadjacency.shape[0]))
index = np.where(counts > 1)[0]
biadjacency = biadjacency[:, index]

stems = list(np.array(list(unique_stems))[index])
pd.DataFrame(stems).to_csv('data/output/names_col.csv', sep=',', index=False, header=False)

df = pd.DataFrame(np.concatenate(
    (biadjacency.nonzero()[0].reshape(-1, 1),
     biadjacency.nonzero()[1].reshape(-1, 1),
     biadjacency.data.reshape(-1, 1)),
    axis=1))
df.columns = ['from', 'to', 'val']
df['from'] = df['from'].astype('long')
df['to'] = df['to'].astype('long')
df['val'] = df['val'].astype('long')
df.to_csv('data/output/biadjacency.csv', index=False, sep=',', header=False)
