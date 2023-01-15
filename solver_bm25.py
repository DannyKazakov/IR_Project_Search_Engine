import data_reader as dr
import numpy as np
import math
import tokenizer
def calculate_BM25_scores(query_to_search, index, folder):
    '''
    calculated the BM25 scores of each candidate document for every token match in the query
    :param query_to_search:
    :param index: the inverted index that contains information on the tokens.
    :param folder: String. which folder to read the posting lists
    :return: dictionary of key - wiki_id and value - bm25 score
    '''
    bm25_scores = {}
    b = 0.75
    k1 = 1.5
    d_avg = sum(index.DL.values()) / len(index.DL)
    tokenized_query = tokenizer.my_tokenizer(query_to_search)
    unique_tokenized_query = np.unique(tokenized_query)
    for query in unique_tokenized_query:
        word, pls = dr.query_word_posting_list(query, index, folder)
        for doc_id, tf in pls:
            doc_idf = math.log(((len(index.DL) - index.df[word] + 0.5) / (index.df[word] + 0.5)) + 1)
            B = 1 - b + b * (index.DL[doc_id] / d_avg)
            bm25_mone = ((k1 + 1) * (tf / index.DL[doc_id])) * doc_idf
            bm25_mechane = B * k1 + (tf / index.DL[doc_id])
            bm25_scores[doc_id] = bm25_scores.get(doc_id, 0) + (bm25_mone / bm25_mechane)
    return bm25_scores
