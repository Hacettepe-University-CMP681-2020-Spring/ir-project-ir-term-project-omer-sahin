import types

from nltk.tokenize import word_tokenize


def tokenize(corpus):
    if type(corpus) is not list:
        return word_tokenize(corpus)

    tokenized_corpus = list()
    for text in corpus:
        tokens = word_tokenize(text)
        tokenized_corpus.append(tokens)

    return tokenized_corpus

