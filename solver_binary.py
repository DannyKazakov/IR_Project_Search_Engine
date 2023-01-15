import tokenizer as tk
import numpy as np
import data_reader as dr


def get_candidate_documents_for_search_title_and_search_anchor(query_to_search, index, id_to_title, folder):
    '''

    :param query_to_search: string
    :param index: the inverted index of the title or anchor
    :param folder: where the posting locs located
    :return: Returns ALL (not just top 100) search results that contain A QUERY WORD
        IN THE TITLE OR IN THE ANCHOR of articles, ordered in descending order of the NUMBER OF
        DISTINCT QUERY WORDS that appear in the title or anchor
    '''
    candidates = {}
    res = []
    tokenized_query = np.unique(tk.default_tokenize(query_to_search))
    for query in tokenized_query:
        word, pls = dr.query_word_posting_list(query, index, folder)
        list_of_doc = [doc_id for doc_id, _ in pls]  # list shell doc id and tf of term doc_id, freq
        for doc_id in list_of_doc:
            candidates[doc_id] = 1 if doc_id not in candidates.keys() else candidates[doc_id] + 1

    if 'anchor' in folder:
        for wiki_id, _ in sorted(candidates.items(), key=lambda x: x[1], reverse=True):
            try:
                title = id_to_title[wiki_id]
                res.append((wiki_id, title))
            except:
                x = 1
    else:
        res = [(wiki_id, id_to_title[wiki_id])
               for wiki_id, _ in sorted(candidates.items(), key=lambda x: x[1], reverse=True)]
    return res

