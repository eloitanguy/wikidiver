# Adapted code by Armand Boschin

import requests
import wikipedia
import os
from bs4 import BeautifulSoup


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


def get_structure():
    """
    Saves the article structure on disk and returns the lists of articles and categories,
    and the dictionary of the main category names
    """

    _articles = []
    _categories = []

    def get_article(article_filename, general_category, sep='|||'):
        """
        Adds the given article to the "articles, categories, general" current structure
        """
        category = []
        with open(article_filename) as _f:
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
                        _articles.append(clean_text(_row))
                        k = 0
                        while _row[k] == '#':
                            k += 1
                        sub_category = sub_category[:k - 1] + [clean_text(_row)]
                        if category[0] == general_category:
                            _categories.append(sep.join(category + sub_category[:-1]))
                        else:
                            _categories.append(sep.join([general_category] + category + sub_category[:-1]))

    categories_dict = get_categories('https://en.wikipedia.org/wiki/Wikipedia:Vital_articles/Level/5')
    _general = {k: v.split('/')[5] for k, v in categories_dict.items()}
    filenames = list(categories_dict.keys())

    if not os.path.exists('wikivitals/data/mds/'):
        os.makedirs('wikivitals/data/mds/')

    for k, v in categories_dict.items():  # saves the category pages' text
        with open('wikivitals/data/mds/{}'.format(k), 'w', encoding='utf8') as f:
            url = "https://en.wikipedia.org/w/index.php?title={}&action=edit".format(v[6:])
            page = requests.get(url)
            soup = BeautifulSoup(page.text, 'html.parser')
            f.write(soup.find('textarea').text)

    for filename in filenames:
        get_article('wikivitals/data/mds/' + filename, _general[filename])

    with open('wikivitals/data/en-categories.txt', 'w', encoding='utf8') as file:
        for cat in _categories:
            file.write(cat + "\n")

    with open('wikivitals/data/en-articles.txt', 'w', encoding='utf8') as file:
        for name in _articles:
            file.write(name + "\n")

    return _articles, _categories, _general


def clean_content_list(text):
    text = clean_text(text)
    text.replace('...', '.')  # we will split sentences using dots
    text.replace('\n', '')
    # remove small sentences (correspond to most headers and blank lines):
    text_sentences_filtered = [sentence for sentence in text.split('.') if len(sentence) > 30]
    return text_sentences_filtered


def get_article_text_by_name(name):
    article = wikipedia.page(name)
    raw = article.content
    return clean_text(raw).replace('\n', '')


def get_article_text_by_page_id(page_id):
    article = wikipedia.page(pageid=page_id)
    raw = article.content
    return clean_text(raw).replace('\n', '')


def get_page_id(name):
    if name[0] == ' ':  # common error caused by a space before the name
        name = name[1:]
    name = name.replace(' ', '%20').replace('\n', '')  # formatting for html query
    url = "https://en.wikipedia.org/w/api.php?action=query&titles={}&format=json".format(name)
    json_response = requests.get(url).json()
    page_id = list(json_response['query']['pages'].keys())[0]
    return page_id  # output first found page id (there should only be one)


class ArticleRetriever(object):
    def __init__(self, article_filename):
        with open(article_filename, 'r') as f:
            self.article_names = [line for line in f.readlines()]

    def save_all_texts(self):
        if not os.path.exists('wikivitals/data/article_texts/'):
            os.makedirs('wikivitals/data/article_texts/')

        for article_idx, name in enumerate(self.article_names):
            page_id = get_page_id(name)
            print('\n', name, page_id)
            # We try to perform disambiguation by going by page ID, if that fails we resort to an imprecise search
            try:
                text = get_article_text_by_page_id(page_id)
            except wikipedia.exceptions.PageError:
                print('Disambiguation error on ', name)
                text = get_article_text_by_name(name)

            with open(os.path.join('wikivitals/data/article_texts/', name + '.txt'), 'w') as f:
                f.writelines(text)


def usa():
    """
    Saves the text of the USA article (for benchmarking)
    """
    if not os.path.exists('wikivitals/data/benchmark/'):
        os.makedirs('wikivitals/data/benchmark/')
    name = 'United States'
    text = get_article_text_by_page_id(get_page_id(name))
    with open(os.path.join('wikivitals/data/benchmark/', name + '.txt'), 'w') as f:
        f.writelines(text)


if __name__ == '__main__':
    # wikipedia.set_lang("en")
    # ar = ArticleRetriever('wikivitals/data/en-articles.txt')
    # ar.save()
    usa()
