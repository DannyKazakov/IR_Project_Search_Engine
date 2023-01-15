from pathlib import Path
import pickle
from google.cloud import storage
from contextlib import closing
# Let's start with a small block size of 30 bytes just to test things out.
BLOCK_SIZE = 1999998
TUPLE_SIZE = 6  # We're going to pack the doc_id and tf values in this
# many bytes.
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer

class MultiFileReader:
    """ Sequential binary reader of multiple files of up to BLOCK_SIZE each. """

    def __init__(self, designated_folder):
        '''

        :param designated_folder: folder contains the posting files of the token we wish to read
        '''
        self._open_files = {}
        self._folder = designated_folder
        self.client = storage.Client()
        self.bucket = self.client.bucket("320569650_bucket")

    def read(self, locs, n_bytes):
        '''
        helper function to read the posting files from disk
        :param locs: list of locations and offset of the file: [('file1.bin, 1000), ('file2.bin', 29844)...]
        :param n_bytes: int. the number of bytes we wish to read from the file
        :return: the object in binary form, which translated to a tupple of (wiki_id, tf) tf - term frequency
        '''
        b = []
        for f_name, offset in locs:
            f_name = f"{self._folder}/" + f_name
            if f_name not in self._open_files:
                blob = self.bucket.blob(f_name)
                # with blob.open("rb") as f:
                self._open_files[f_name] = blob.open("rb")
            f = self._open_files[f_name]
            f.seek(offset)
            n_read = min(n_bytes, BLOCK_SIZE - offset)
            b.append(f.read(n_read))
            n_bytes -= n_read
        return b''.join(b)

    def close(self):
        for f in self._open_files.values():
            f.close()

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False


def query_word_posting_list(query_word, index, folder):
    '''

    :param query_word: String. the token we want to read it's posting list
    :param index: InvertedIndex object. hold the posting locs of the tokenized text and we use it to read the token's postings
    :param folder: String. from which folder in the bucket we read the posting lists
    :return: returns the query_word, and a list of posting list in the form og : [(wiki_id, tf)......]
    '''

    with closing(MultiFileReader(folder)) as reader:
        posting_list = []
        if query_word in index.posting_locs.keys():
            locs = index.posting_locs[query_word] # [('87_012.bin', 1843674), ('87_013.bin', 0), ('87_014.bin', 0)]y
            b = reader.read(locs, index.df[query_word] * TUPLE_SIZE)
            for i in range(index.df[query_word]):
                doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
                tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
                posting_list.append((doc_id, tf))
        return query_word, posting_list

def read_dict(base_dir, name):
    '''
    this function reads the pickled dictionary and load it to a Dictionary object
    used for reading the ".dic" files which hold the id2title dictionary as well as the norms of the body and title
    :param base_dir: string. the folder from which we read the dictionary
    :param name: String. the name of the file we wish to read
    :return: the pickled loaded file we read
    '''
    with open(Path(base_dir) / f'{name}.dic', 'rb') as f:
        return pickle.load(f)


def read_index(base_dir, name):
    '''
    this function reads the pickled InvertedIndex and load it to a InvertedIndex object
    used for reading the ".pkl" files which are the indices of the body and title
    :param base_dir: string. the folder from which we read the dictionary
    :param name: String. the name of the file we wish to read
    :return: the pickled loaded file we read
    '''
    with open(Path(base_dir) / f'{name}.pkl', 'rb') as f:
        return pickle.load(f)


