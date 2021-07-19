import pandas as pd
from textblob import TextBlob
from textblob.translate import Translator


def get_language(tweet):
    return TextBlob(tweet).detect_language()

def translate_to_eng(paragraph):
    paragraph = str(paragraph).strip().split('.')
    translated = []

    for sentence in paragraph:
        try:
            sleep(1.0)
            sentence = sentence.strip()
            en_blob = Translator()
            translated.append(str(en_blob.translate(sentence, from_lang=get_language(sentence), to_lang='en')))
            sleep(1.0)
        except Exception as e:
            print(e)

    return translated

def remove_enter(row):
    row = row.replace('\n', ' ')
    return row























