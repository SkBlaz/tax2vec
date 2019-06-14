
## routines for text preprocessing!
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer, TfidfTransformer
from sklearn.metrics import precision_recall_fscore_support
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.sparse import hstack
import gzip
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn import preprocessing
import nltk
import numpy as np
import pandas as pd
from nltk import word_tokenize
from nltk.corpus import stopwords
import re
import string
from itertools import groupby
try:
    from nltk.tag import PerceptronTagger
except:
    def PerceptronTagger():
        return 0
    
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import FeatureUnion
from sklearn import pipeline
from sklearn.preprocessing import Normalizer

sent_tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')
#perceptron_tagger = PerceptronTagger()

def remove_punctuation(text):
    table = text.maketrans({key: None for key in string.punctuation})
    text = text.translate(table)
    return text


def remove_stopwords(text):
    stops = set(stopwords.words("english"))
    text = text.split()
    text = [x.lower() for x in text if x.lower() not in stops]
    return " ".join(text)

def remove_stopwords_and_articles(text):
    stops = set(stopwords.words("english"))
    stops.add("less")
    text = text.split()
    text = [x.lower() for x in text if x.lower() not in stops and len(x) > 2]
    return " ".join(text)

def remove_mentions(text, replace_token):
    return re.sub(r'(?:@[\w_]+)', replace_token, text)


def remove_hashtags(text, replace_token):
    return re.sub(r"(?:\#+[\w_]+[\w\'_\-]*[\w_]+)", replace_token, text)


def remove_url(text, replace_token):
    regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    return re.sub(regex, replace_token, text)


# def tag(tagger, text, sent_tokenize):
#     #tokenize with nltk default tokenizer
#     tokens = sent_tokenize.tokenize(str(text))
#     #use average perceptron tagger
#     tokens = [word_tokenize(token) for token in tokens]
#     text = tagger.tag_sents(tokens)
#     return " ".join(tag for sent in text for word, tag in sent)

def simplify_tag(t):
    if "+" in t:
        return t[t.index("+")+1:]
    else:
        return t

def get_emojis(path):
    emoji_dict = {}
    try:
        df_emojis = pd.read_csv(path, encoding="utf-8", delimiter=",")
        for index, row in df_emojis.iterrows():
            occurrences = row['Occurrences']
            pos = (row['Positive'] + 1) / (occurrences + 3)
            neg = (row['Negative'] + 1) / (occurrences + 3)
            sent = pos - neg
            emoji_dict[row['Emoji']] = sent
    except:
        emoji_dict = {}
    return emoji_dict


def countCharacterFlooding(text):
    text = ''.join(text.split())
    groups = groupby(text)
    cnt = 0
    for label, group in groups:
        char_cnt = sum(1 for _ in group)
        if char_cnt > 2:
            cnt += 1
    return cnt

#count words in tweet that are in a specific word list and return frequency
def countWords(wordList, text):
    cnt = 0
    length = len(text.split())
    for word in text.split():
        if word.lower() in wordList:
            cnt +=1
    if length == 0:
        return 0
    return cnt/length

#count specific characters
def count_patterns(text, list):
    cnt=0
    length = len(text)
    for pattern in list:
        cnt += text.count(pattern)
    if length == 0:
        return 0
    return cnt/length

#get sentiment according to emojis
def get_sentiment(text, emoji_dict):
    sentiment = 0
    list = emoji_dict.keys()
    for pattern in list:
        text_cnt = text.count(pattern)
        sentiment += emoji_dict[pattern] * text_cnt
    return sentiment


#count specific pos tags and return frequency
def count_pos(pos_sequence, pos_list):
    cnt = 0
    for pos_tag in pos_sequence.split():
        for pos in pos_list:
            if pos_tag == pos:
                cnt += 1
    return cnt/len(pos_sequence.split())


def get_affix(text):
    return " ".join([word[-4:] if len(word) >= 4 else word for word in text.split()])


def affix_punct(text):
    punct = '!"$%&()*+,-./:;<=>?[\]^_`{|}~'
    ngrams = []
    for i, character in enumerate(text[0:-2]):
        ngram = text[i:i+3]
        if ngram[0]  in punct:
            for p in punct:
                if p in ngram[1:]:
                    break
            else:
                ngrams.append(ngram)
    return "###".join(ngrams)

def affix_punct_tokenize(text):
    tokens = text.split('###')
    return tokens


class text_col(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    def fit(self, x, y=None):
        return self
    def transform(self, data_dict):
        return data_dict[self.key]


#fit and transform numeric features, used in scikit Feature union
class digit_col(BaseEstimator, TransformerMixin):
    def fit(self, x, y=None):
        return self
    def transform(self, hd_searches):
        d_col_drops=['text', 'no_punctuation', 'no_stopwords', 'text_clean', 'affixes', 'affix_punct']
        hd_searches = hd_searches.drop(d_col_drops, axis=1).values
        scaler = preprocessing.MinMaxScaler().fit(hd_searches)
        return scaler.transform(hd_searches)


def build_dataframe(data_docs):
    df_data = pd.DataFrame({'text': data_docs})
    df_data['text_clean_r'] = df_data['text'].map(lambda x: remove_hashtags(x, '#HASHTAG'))
    df_data['text_clean_r'] = df_data['text_clean_r'].map(lambda x: remove_url(x, "HTTPURL"))
    df_data['text_clean_r'] = df_data['text_clean_r'].map(lambda x: remove_mentions(x, '@MENTION'))
    df_data['text_clean'] = df_data['text'].map(lambda x: remove_hashtags(x, ''))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_url(x, ""))
    df_data['text_clean'] = df_data['text_clean'].map(lambda x: remove_mentions(x, ''))

#    df_data['pos_tag'] = df_data['text_clean'].map(lambda x: tag(perceptron_tagger, x, sent_tokenizer))
    df_data['no_punctuation'] = df_data['text_clean'].map(lambda x: remove_punctuation(x))
    df_data['no_stopwords'] = df_data['no_punctuation'].map(lambda x: remove_stopwords(x))
    df_data['text_clean'] = df_data['text_clean_r']
    df_data = df_data.drop('text_clean_r', 1)
    emoji_dict = get_emojis('resources/Emoji_Sentiment_data_v1.0.csv')
    emoji_list = emoji_dict.keys()
    df_data['affixes'] = df_data['text_clean'].map(lambda x: get_affix(x))
    df_data['affix_punct'] = df_data['text_clean'].map(lambda x: affix_punct(x))
    df_data['number_of_emojis'] = df_data['text_clean'].map(lambda x: count_patterns(x, emoji_list))
    df_data['sentiment'] = df_data['text_clean'].map(lambda x: get_sentiment(x, emoji_dict))
    df_data['number_of_character_floods'] = df_data['no_punctuation'].map(lambda x: countCharacterFlooding(x))
    return df_data

def data_docs_to_matrix(data_docs, mode="count",max_features = 10000, ngram_range=(2,6), simple_clean=False):

    if simple_clean:
        data_docs = [remove_stopwords_and_articles(x) for x in data_docs]
    
    if mode == "matrix_word":
        tokenizer = TfidfVectorizer(stop_words="english",strip_accents="ascii",analyzer="word",max_features=max_features, ngram_range=ngram_range)
        tokenizer = tokenizer.fit(data_docs)
        encoded_docs = tokenizer.transform(data_docs)
        return (encoded_docs,tokenizer, None)

    if mode == "matrix_char":
        tokenizer = TfidfVectorizer(stop_words="english",strip_accents="ascii",analyzer="char",max_features=max_features, ngram_range=ngram_range)
        tokenizer = tokenizer.fit(data_docs)
        encoded_docs = tokenizer.transform(data_docs)
        return (encoded_docs,tokenizer, None)

    if mode == "matrix_pan":
        df_data = build_dataframe(data_docs)

        tfidf_unigram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=10, max_df=0.8)
        tfidf_bigram = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=False, min_df=20, max_df=0.5)
        #tfidf_pos = TfidfVectorizer(ngram_range=(2, 2), sublinear_tf=True, min_df=0.1, max_df=0.6, lowercase=False)
        character_vectorizer = CountVectorizer(analyzer='char_wb', ngram_range=(4, 4), lowercase=False, min_df=4,max_df=0.8)
        tfidf_ngram = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=0.1, max_df=0.8)
        tfidf_transformer = TfidfTransformer(sublinear_tf=True)
        tfidf_affix_punct = TfidfVectorizer(ngram_range=(1, 1), sublinear_tf=True, min_df=0.1, max_df=0.8, tokenizer=affix_punct_tokenize)

        features = [('cst', digit_col()),
                    ('unigram', pipeline.Pipeline([('s1', text_col(key='no_stopwords')), ('tfidf_unigram', tfidf_unigram)])),
                    ('bigram', pipeline.Pipeline([('s2', text_col(key='no_punctuation')), ('tfidf_bigram', tfidf_bigram)])),
 #                   ('tag', pipeline.Pipeline([('s4', text_col(key='pos_tag')), ('tfidf_pos', tfidf_pos)])),
                    ('character', pipeline.Pipeline(
                        [('s5', text_col(key='text_clean')), ('character_vectorizer', character_vectorizer),
                         ('tfidf_character', tfidf_transformer)])),
                    ('affixes', pipeline.Pipeline([('s5', text_col(key='affixes')), ('tfidf_ngram', tfidf_ngram)])),
                    ('affix_punct', pipeline.Pipeline(
                        [('s5', text_col(key='affix_punct')), ('tfidf_affix_punct', tfidf_affix_punct)])),
                    ]
        weights = {'cst': 0.3,
                   'unigram': 0.8,
                   'bigram': 0.1,
  #                 'tag': 0.2,
                   'character': 0.8,
                   'affixes': 0.4,
                   'affix_punct': 0.1,
                   }

        matrix = pipeline.Pipeline([
            ('union', FeatureUnion(
                transformer_list=features,
                transformer_weights=weights,
                n_jobs=1
            )),
            ('scale', Normalizer())])


        tokenizer = matrix.fit(df_data)
        print(df_data.shape, df_data.columns)
        encoded_docs = tokenizer.transform(df_data)
        print('Matrix shape: ', encoded_docs.shape)

        return (encoded_docs, tokenizer, None)

    if mode == "index_char":
        
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=' ', char_level=True, oov_token=None)

        tokenizer.fit_on_texts(data_docs)
        sequences = tokenizer.texts_to_sequences(data_docs)
        maxlen = np.max([len(x) for x in sequences])
        padded_docs = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')
        
        return (padded_docs, tokenizer, maxlen)
                
    if mode == "index_word":
        tokenizer = tf.keras.preprocessing.text.Tokenizer(num_words=None, filters='!"#$%&()*+,-./:;<=>?@[\]^_`{|}~', lower=True, split=" ", char_level=False, oov_token=None)

        tokenizer.fit_on_texts(data_docs)
        
        sequences = tokenizer.texts_to_sequences(data_docs)
        maxlen = np.max([len(x) for x in sequences])
        padded_docs = tf.keras.preprocessing.sequence.pad_sequences(sequences, maxlen=maxlen, padding='post')

        return (padded_docs, tokenizer, maxlen)


def check_requirements():

    ## get wordnet if not present
    try:
        import ssl
        
        try:
            _create_unverified_https_context = ssl._create_unverified_context
        except AttributeError:
            pass
        else:
            ssl._create_default_https_context = _create_unverified_https_context
        nltk.download("wordnet")
    except:
        pass

def generate_train_test_split(trainfile,max_docs=10):

    labels_train,dcorp_train,multilabel = read_csv_gz(trainfile,max_docs=max_docs)
    labels_test,dcorp_test,multilabel = read_csv_gz(trainfile.replace("train","test"),max_docs=max_docs)

    labels = labels_train+labels_test
    onehot_encoder = OneHotEncoder(sparse=False,categories="auto")
    
    if multilabel:
        mlb = MultiLabelBinarizer()
        onehot_encoder = mlb.fit(labels)
        
    else:
        label_encoder = LabelEncoder()  
        integer_encoded = label_encoder.fit_transform(labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        labels = onehot_encoder.fit_transform(integer_encoded)

    labels_train = labels[0:len(labels_train)]
    labels_test = labels[len(labels_train):]                          
    return (dcorp_train,dcorp_test,labels_train,labels_test)        

def read_csv_gz(dfile,max_docs=10):
    labels = []
    dcorp = []
    multilabel = False
    if ".csv.gz" in dfile:
        with gzip.open(dfile,"r") as df:
            for line in df:
                splits = line.decode().strip().split(",")
                ## zero is the label                
                label = splits[0]
                if label != "type" and label != "rating":
                    if "&&&" in label:
                        multilabel = True
                        labels.append(label.split("&&&"))
                    else:
                        labels.append(label)

                    ## from 1 on are the documents
                    docs = " MERGERTAG ".join(",".join(splits[1:]).split("|||")[0:max_docs])
                    dcorp.append(docs)
                    
    return (labels,dcorp,multilabel)

def generate_corpus(dfile,max_docs=10):
    
    labels,dcorp,multilabel = read_csv_gz(dfile,max_docs=max_docs)                    
    print("Number of target labels {}".format(len(set(labels))))
    onehot_encoder = OneHotEncoder(sparse=False)
    if multilabel:
        mlb = MultiLabelBinarizer()
        onehot_encoded = mlb.fit_transform(labels)
    else:
        label_encoder = LabelEncoder()  
        integer_encoded = label_encoder.fit_transform(labels)
        integer_encoded = integer_encoded.reshape(len(integer_encoded), 1)
        onehot_encoded = onehot_encoder.fit_transform(integer_encoded)
    class_names = list(label_encoder.classes_)
    return (onehot_encoded,dcorp,class_names)

def split_generator(matrix,corpus,labels,num_splits=3,test=0.1):

    ## generate stratified splits..
    sss = StratifiedShuffleSplit(n_splits=num_splits, test_size=test, random_state=0)
    for train_index, test_index in sss.split(matrix, labels):
        train_x = [corpus[i] for i in train_index]
        test_x = [corpus[i] for i in test_index]
        train_y = labels[train_index]
        test_y = labels[test_index]
        yield (train_x,test_x,train_y,test_y)
