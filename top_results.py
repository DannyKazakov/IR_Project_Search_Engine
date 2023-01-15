def get_top_n_results(sim_dict,id2title, N=3):
    """
    Sort and return the highest N documents according to the cosine similarity score.
    Generate a dictionary of cosine similarity scores

    Parameters:
    -----------
    sim_dict: a dictionary of similarity score as follows:
                                                                key: document id (e.g., doc_id)
                                                                value: similarity score. We keep up to 5 digits after the decimal point. (e.g., round(score,5))

    N: Integer (how many documents to retrieve). By default N = 3

    Returns:
    -----------
    a ranked list of pairs (doc_id, score) in the length of N.
    """
    id_and_scores = [(wiki_id, score) for wiki_id, score in sim_dict.items()]
    sorted_id_scores = sorted(id_and_scores, key=lambda x: x[1], reverse=True)[:N]
    return [(int(wiki_id), id2title[wiki_id]) for wiki_id,_ in sorted_id_scores]


def merge_results_top_results(id_to_title, title_scores, body_scores, title_weight=0.5, body_weight=0.5, N = 3):
  '''
  title_scores - key - doc_id value bm25score
  body_scores - key - doc_id value bm25score
  '''
  all_docs = set(title_scores.keys()).union(body_scores.keys()) # all keys
  merged_score = {}
  for doc_id in all_docs:
      merged_score[doc_id] = title_scores.get(doc_id, 0) * title_weight + body_scores.get(doc_id, 0) * body_weight
  return get_top_n_results(merged_score, id_to_title, N)
