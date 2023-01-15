import math
from collections import Counter
import numpy as np

import data_reader as dr
import tokenizer
import top_results


def generate_query_tfidf_vector(query_to_search, index):
    """
    Generate a vector representing the query. Each entry within this vector represents a tfidf score.
    The terms representing the query will be the unique terms in the index.

    We will use tfidf on the query as well.
    For calculation of IDF, use log with base 10.
    tf will be normalized based on the length of the query.

    Parameters:
    -----------
    query_to_search: list of tokens (str). This list will be preprocessed in advance (e.g., lower case, filtering stopwords, etc.').
                     Example: 'Hello, I love information retrival' --->  ['hello','love','information','retrieval']

    index:           inverted index loaded from the corresponding files.

    Returns:
    -----------
    vectorized query with tfidf scores
    """
    epsilon = .0000001
    total_vocabulary_size = len(query_to_search)
    Q = np.zeros(total_vocabulary_size)
    term_vector = list(query_to_search)
    counter = Counter(query_to_search)
    for token in np.unique(query_to_search):
        if token in index.df.keys():  # avoid terms that do not appear in the index.
            tf = counter[token] / len(query_to_search)  # term frequency divded by the length of the query
            df = index.df[token]
            idf = math.log((len(index.DL)) / (df + epsilon), 10)  # smoothing

            try:
                ind = term_vector.index(token)
                Q[ind] = tf * idf
            except:
                pass
    return Q


def calculate_tfidf_scores_for_unique_candidates(query_to_search, index, folder):
    '''
    this function calculates the tfidf scores of each wiki page candidate for a token in the query.
    :param query_to_search: List of strings. each item is a token derived from the query we got from the user
    :param index: the inverted index which contains needed information about the tokens
    :param folder: string. from which folder we want to read the posting files of a token in query_to_search
    :return: candidates_scores - key = (wiki_id, token) value - tfidf score of the token in the wikipedia page
             unique_candidates - a set of unique wiki_id documents which were founs in the posting list of the token
             matched_token - a list of token we matched with the query_to_search after when trying to read them from the
             posting lists
    '''
    candidates = {}
    unique_candidates = set()
    matched_tokens = []
    for query in query_to_search:
        word, pls = dr.query_word_posting_list(query, index, folder)
        for doc_id, tf in pls:
            if doc_id not in unique_candidates:
                unique_candidates.add(doc_id)
            candidates[(doc_id, word)] = (tf / index.DL[doc_id]) * math.log(len(index.DL) / index.df[word], 10)
        matched_tokens.append(word)
    return candidates, unique_candidates, matched_tokens


def calculate_cosine_similarity_for_each_candidate(query, index, folder, vector_norm, use_default_tokenizer):
    '''
    This function calculates the cosine similarity of the query with each wiki page candidate
    :param query: string. the query of the user
    :param index: inverted index file
    :param folder: which folddr to read the posting file of  query
    :param vector_norm: dictionary of key - wiki_id and value - the calculated norm of the vectorized doc_id. int
    :param use_default_tokenizer: boolean. to use the default tokenizer or our custom tokenizer
    :return: dictionary of key - wiki_id and value - cosine similarity score
    '''
    if use_default_tokenizer:
        query_to_search = np.unique(tokenizer.default_tokenize(query))
    else:
        query_to_search = np.unique(tokenizer.my_tokenizer(query))

    query_size = len(query_to_search)  # only the terms in the query in the corpus
    term_vector = list(query_to_search)  # each item is token in the corpus
    cosine_similarity_dict = {}

    candidates_scores, unique_candidates, matched_tokens = \
        calculate_tfidf_scores_for_unique_candidates(query_to_search, index, folder)
    query_vector = generate_query_tfidf_vector(query_to_search, index)
    # go after each document in the unique candidates and calculate it's cosine similarity with the query
    for doc_id in unique_candidates:
        doc_id_vector = np.zeros(query_size)
        for token in matched_tokens:
            index_of_query_word = term_vector.index(token)
            doc_id_vector[index_of_query_word] = candidates_scores.get((doc_id, token), 0)
        cosine_similarity_dict[doc_id] = np.dot(doc_id_vector, query_vector) / (
                    (vector_norm[doc_id]) * np.linalg.norm(query_vector))
    return cosine_similarity_dict


def get_results_for_body(query, body_index, id_to_norm_body, id2title, folder):
    '''
    This function gets the top 100 results for the query looked up in the body of the wikipedia corpus.
    :param query: String. the query of the user
    :param body_index: the Inverted index of the body
    :param id_to_norm_body: dictionary of wiki_id and the calculated norm of the wiki_id
    :param id2title: dictionary of int wiki_id to String title
    :param folder: String. the folder that contains the posting files
    :return: list of top 100 wiki_id, title of the query in the body. [(wiki_id, title),....]
    '''
    wiki_id_to_scores = calculate_cosine_similarity_for_each_candidate(query, body_index, folder, id_to_norm_body, True)
    return top_results.get_top_n_results(wiki_id_to_scores, id2title, 100)


def get_results_for_index(query, body_index, id_to_norm_body, body_folder, title_index, id_to_norm_title, title_folder,
                          id2title):
    wiki_id_body_scores = calculate_cosine_similarity_for_each_candidate(query, body_index,
                                                                         body_folder, id_to_norm_body, False)
    wiki_id_title_scores = calculate_cosine_similarity_for_each_candidate(query, title_index,
                                                                          title_folder, id_to_norm_title, False)
    return top_results.merge_results_top_results(id2title, wiki_id_title_scores, wiki_id_body_scores, N=100)
