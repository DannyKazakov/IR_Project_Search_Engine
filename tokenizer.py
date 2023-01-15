import re
import nltk
from nltk.stem.porter import *
from nltk.corpus import stopwords
nltk.download('stopwords')
################################################### STOP WORDS #########################################################
english_stopwords = frozenset(stopwords.words('english'))
############################################## My Regular Expressions ##################################################
# YOUR CODE HERE
def get_html_pattern():
  '''
  :return: regular expression for html tags
  '''
  return "<(.*?)>"
def get_date_pattern():
  '''
  :return: regular expression for dates
  '''
  months31A = "([1-9]\s|1[0-9]\s|2[0-9]\s|3[0-1]\s)(jan|mar|may|july|aug|oct|dec)"
  months31B = "(jan\s|mar\s|may\s|july\s|aug\s|oct\s|dec\s)([1-9],?|1[0-9],?|2[0-9],?|3[0-1],?)"
  months31C = "(january\s|march\s|may\s|july\s|august\s|october\s|december\s)([1-9],?|1[0-9],?|2[0-9],?|3[0-1],?)"

  months30A = "([1-9]\s|1[0-9]\s|2[0-9]\s|30\s)(nov|sep|june|apr)"
  months30B = "(nov\s|sep\s|june\s|apr\s)([1-9],?|1[0-9],?|2[0-9],?|30,?)"
  months30C = "(november\s|september\s|june\s|april\s)([1-9],?|1[0-9],?|2[0-9],?|30,?)"

  februaryA = "([1-9]\s|1[0-9]\s|2[0-9]\s)feb"
  februaryB = "feb\s([1-9],?|1[0-9],?|2[0-9],?)"
  februaryC = "(february\s)([1-9],?|1[0-9],?|2[0-9],?)"

  year = "\s([1-9][0-9]{0,3})"
  dateExpresssion = "(({0}|{1}|{2})|({3}|{4}|{5})|({6}|{7}|{8})){9}"
  return dateExpresssion.format(months31A, months31B, months31C, months30A, months30B, months30C, februaryA, februaryB, februaryC, year)
def get_time_pattern():
  '''
  :return: regular expression for time
  '''
  time_pattern_one  = r"(\b(?<!\S)(1[0-9]|2[0-3]|[0-9])(:[0-5][0-9]){2}\b)"
  time_pattern_two = r"(\b([0-9]|1[0-1])\.([0-5][0-9])((AM|PM)\b))|(\b(([1-9])|1[0-2])\.([0-5][0-9])((AM|PM)\b))"
  time_pattern_three = r"(\b(0[0-9]|1[0-2])([0-5][0-9])(a\.m\.|p\.m\.)(?![\w\d]))"
  return "({0})|({1})|({2})".format(time_pattern_one, time_pattern_two, time_pattern_three)

def get_percent_pattern():
  '''
  :return: regular expression for percent
  '''
  return "([1-9]+\.?\d*%)|([0-9]{1}\.[0-9]*)%|(0%)"
def get_number_pattern():
  '''
  :return: regular expression numbers
  '''
  return r"((?<![\-\+0-9a-z\,\.])[\-\+]?\b[0-9]{1,3}(\,[0-9]{3})*(\.[0-9]+)?\b(?!\,\d|\,\w|\.\d|\.\w|\%))"
def get_word_pattern():
  '''
  :return: regular expression for words
  '''
  return r"((?<![\-a-zA-Z0-9])\w*(\w+\'?\w?(\-\w+)*)(?![\-a-zA-Z0-9]))"
########################################################################################################################
########################################################################################################################

################################################## Tokenizers ##########################################################
def default_tokenize(text):
    '''

    :param text: String representing the text to tokenize
    :return: list of tokens generated by the tokenize function with the given tokenizer
    '''
    RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)
    return tokenize(text, RE_WORD)

def my_tokenizer(text):
    RE_TOKENIZE = re.compile(rf"""
    (
    # parsing html
    (?P<HTMLTAG>{get_html_pattern()})
    #dates
    | (?P<DATE>{get_date_pattern()})
    #time
    | (?P<TIME>{get_time_pattern()})
    #percents
    | (?P<PERCENT>{get_percent_pattern()})
    #Numbers
    | (?P<NUMBER>{get_number_pattern()})
    #Words
    | (?P<WORD>{get_word_pattern()})
    #Space
    | (?P<SPACE>[\s\t\n]+)
    #else
    | (?P<OTHER>.))
    """, re.MULTILINE | re.IGNORECASE | re.VERBOSE | re.UNICODE)
    return tokenize(text, RE_TOKENIZE)


def tokenize(text, tokenizer):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filters stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.
    tokenizer: re object represents the method of tokenizing the text
    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """

    list_of_tokens = [token.group() for token in tokenizer.finditer(text.lower()) if
                      token.group() not in english_stopwords]
    return list_of_tokens
