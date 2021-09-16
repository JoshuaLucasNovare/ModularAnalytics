import pandas as pd
from textblob import TextBlob
from textblob.translate import Translator


def get_language(tweet):
    return TextBlob(tweet).detect_language()

def translate_to_eng(text):
    """Translates text into the target language.
    
    Target must be an ISO 639-1 language code.
    See https://g.co/cloud/translate/v2/translate-reference#supported_languages
    """
    import six
    from google.cloud import translate_v2 as translate

    translate_client = translate.Client()

    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    
    # Text can also be a sequence of strings, in which case this method
    # will return a sequence of results
    result = translate_client.translate(text, target_language='en')

    return [result['translatedText']]

def remove_enter(row):
    row = row.replace('\n', ' ')
    return row























