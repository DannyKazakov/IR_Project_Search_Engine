# SearchEngineOnWikipedia
Search engine project for IR course 2022, Daniel Kazakov and Itay Paikin.

## Intro
During our 3rd year in the Information and Software System Engineering studies, we took a course at Information Retrieval.
In that course, we gained knowledge regarding different retrieval, indexing, crawling and evaluation techniques, which resulted in creating the following project.
The project ran during 2 days on the gcp (Google cloud) compute engine,
and answered queries through 5 different URL's - retrieving pages from the entire english wikipedia corpus.

## Content
Contains all the files needed to be in the computer engine instance in order to run the search engine without fail.
* data_reader.py - helper in order to read from the posting lists contained in the GCP bucket
* engine_startup.py - downloads necessary files from the bucket in to the instance. indices, dictionaries
* search_frontend.py - runs the Flask RESFul API and make the engine public for queries
* solver_binary.py - rates the retrieved documents based on distinct occurrences of a word from the query in text
* solver_bm25.py - contains the method of implementing the bm25 algorithm for docs and query similarity.
* solver_cosine_similarity.py - contains methods of implementing the cosine similarity for docs for a given query
* tokenizer.py - contains different tokenizing options for the query
* top_results.py - contains method of returning the top 100 results and for merging results from diffrenet indices
## Capabillities
Through the engines end points, you can retrieve information using 5 different techniques:

* Search: retrive information with a query, use both body and title index (0.50 -0.25 ratio). 
* Search body: retrive information only through the wiki page body. Use tf-idf and cosine similarity measure for comparison.
* Serach title: retrive information only through the wiki page title. Use a binary ranking of terms existing or not in the title. More terms in title get prioritized.
* Search anchor: retrive information only through the wiki page title. Use a binary ranking of terms existing or not in the title. More terms in title get prioritized.
* get_pagerank: retrive a specific wiki page rank.
* get_pageview: retrive a specific wiki page amount of views.
## Endpoints
Our search engine supports 5 different requests, including:

* [GET] request, route: /search. Insert your query through the 'query' parameter.
* [GET] request, route: /search_body. Insert your query through the 'query' parameter.
* [GET] request, route: /search_title. Insert your query through the 'query' parameter.
* [GET] request, route: /search_anchor. Insert your query through the 'query' parameter.
* [POST] request, route: /get_pagerank. Insert wiki id's in the body of the request in a parameter named 'json'.
* [POST] request, route: /get_pageview. Insert wiki id's in the body of the request in a parameter named 'json'.

## Evaluation
We evaluated our engine using MAP@40. The results reached an average of 0.467 at the submission of the project.
Retrieval time was 7.54 seconds on average.
![image](https://user-images.githubusercontent.com/81555212/212553996-ef65d409-3be8-40e5-b6f6-e4e3f767a097.png)

## References
* Python packages - including pickle, json, nltk and flask.
* Google storage, a link to our project bucket: https://console.cloud.google.com/storage/browser/320569650_bucket
* Virtual machine external IP: http://34.66.155.185:8080 , can be activated and queried through /search?query=YOUR_QUERY ! Email us for activating the VM.

Daniel Kazakov: kazadan@post.bgu.ac.il 
Itay Paikin: itaypai@post.bgu.ac.il
