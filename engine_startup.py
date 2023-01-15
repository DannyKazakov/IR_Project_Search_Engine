from google.cloud import storage

# get the GCS
bucket_name = '320569650_bucket'
storage_client = storage.Client()
bucket_dani = storage_client.bucket(bucket_name)


def get_indices_to_webserver():
    '''
    This function downloads the inverted indices from the GCP bucket
    :return: doesn't return anything
    '''
    blob_body = bucket_dani.blob("body_index/index.pkl")
    blob_body_regex = bucket_dani.blob("my_regex_index/regex_index.pkl")
    blob_title = bucket_dani.blob("title_index_org/title_index_org.pkl")
    blob_title_regex = bucket_dani.blob("title_regex/title_regex.pkl")
    blob_anchor = bucket_dani.blob("anchor_index/anchor.pkl")

    blob_body.download_to_filename("body_index.pkl")
    blob_body_regex.download_to_filename("my_regex_index.pkl")
    blob_title.download_to_filename("title_index_org.pkl")
    blob_title_regex.download_to_filename("title_regex.pkl")
    blob_anchor.download_to_filename("anchor_index.pkl")


def get_id2dictionary():
    '''
    This function downloads the dictionaries of the norms of the title indices and body indices
    as well as downloading the id to title dictionary to the instance
    :return: doesn't return anything
    '''
    blob_id2dic = bucket_dani.blob("id2title.dic")  # dictionary key - doc_id value - doc title
    blob_id2dic.download_to_filename("id2title.dic")

    blob_id2norm = bucket_dani.blob("body_index/id2norm.dic")
    blob_id2norm_title = bucket_dani.blob("id2norm_title.dic")
    blob_id2norm_regex = bucket_dani.blob("id2norm_regex.dic")
    blob_id2norm_title_regex = bucket_dani.blob("id2norm_title_regex.dic")

    blob_id2norm.download_to_filename("id2norm.dic")
    blob_id2norm_regex.download_to_filename("id2norm_regex.dic")
    blob_id2norm_title.download_to_filename("id2norm_title.dic")
    blob_id2norm_title_regex.download_to_filename("id2norm_title_regex.dic")


# get the inverted index module from the bucket

def get_packages_from_bucket():
    '''
    this function downloads the inverted_index_gcp which contains the inverted index class
    downloads the pagerank file
    downloads the pageviews file of the users in august 2021
    :return: doesn't return anything
    '''
    blob = bucket_dani.blob("inverted_index_gcp.py")
    blob.download_to_filename("inverted_index_gcp.py")
    blob = bucket_dani.blob("pr/part-00000-4c134cf7-de3e-4ca6-865e-1b1d7986f869-c000.csv.gz")
    blob.download_to_filename("pagerank.csv.gz")
    blob = bucket_dani.blob("pageviews-202108-user.pkl")
    blob.download_to_filename("pageviews-202108-user.pkl")


def init_engine():
    '''
    calls all the function above
    :return: doesn't return anything
    '''
    print("Downloading files from bucket..")
    get_indices_to_webserver()
    get_id2dictionary()
    get_packages_from_bucket()
