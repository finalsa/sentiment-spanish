import _pickle as pickle
import bz2
import re
from pathlib import Path

REPLACE_BY_SPACE_RE = re.compile('[/(){}\[\]\|@,;]')
BAD_SYMBOLS_RE = re.compile('[^\w\s]')


def compressed_pickle(title, data):
    with bz2.BZ2File(title + '.pbz2', 'w') as f:
        pickle.dump(data, f)


def decompress_pickle(file):
    data = bz2.BZ2File(file, 'rb')
    data = pickle.load(data)
    return data


def clean_text(text):
    text = text.lower()
    text = REPLACE_BY_SPACE_RE.sub(' ', text)
    text = BAD_SYMBOLS_RE.sub('', text)
    return text


class SentimentAnalysisSpanish:
    def __init__(self):
        self.parent_path = Path(__file__).parent
        path_vectorizer = str(self.parent_path / 'saved_model/ngram_vectorized_compressed.pbz2')
        self.vectorizer = decompress_pickle(path_vectorizer)
        path_classifier = str(self.parent_path / 'saved_model/classifier_naive_bayes_compressed.pbz2')
        self.classifier = decompress_pickle(path_classifier)


    # returns the sentiment of a text string
    def sentiment(self, text: str):
        vals = self.vectorizer.transform([text])
        return self.classifier.predict_proba(vals)[0][1]
