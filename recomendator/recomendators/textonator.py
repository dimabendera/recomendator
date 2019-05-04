import pymysql.cursors
import json
import sys
import gensim
from gensim import corpora
from pprint import pprint
from gensim import models
import numpy as np
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords
import re
import logging
import pymorphy2
from stop_words import get_stop_words
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import re
from tqdm import tqdm
from string import punctuation
import string
import keras
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional, concatenate
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import json
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=False,
                        device_count = {'CPU': 1})
session = tf.Session(config=config)
keras.backend.set_session(session)


import pymysql.cursors
import json
import sys
import gensim
from gensim import corpora
from pprint import pprint
from gensim import models
import numpy as np
from gensim.models import LdaModel, LdaMulticore
import gensim.downloader as api
from gensim.utils import simple_preprocess, lemmatize
from nltk.corpus import stopwords
import re
import logging
import pymorphy2
from stop_words import get_stop_words
from gensim.models.word2vec import Word2Vec
from multiprocessing import cpu_count
import gensim.downloader as api
from nltk.tokenize import WordPunctTokenizer
from collections import Counter
from string import punctuation, ascii_lowercase
import re
from tqdm import tqdm
from string import punctuation
import string
import keras
from keras.layers import Dense, Input, LSTM, Embedding, Dropout,SpatialDropout1D, Bidirectional
from keras.models import Model
from keras.optimizers import Adam
from keras.layers import BatchNormalization
from keras.preprocessing.sequence import pad_sequences
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.base import BaseEstimator, ClassifierMixin
import os
import json
from sklearn.model_selection import GridSearchCV
import tensorflow as tf

config = tf.ConfigProto(intra_op_parallelism_threads=1,
                        inter_op_parallelism_threads=1,
                        allow_soft_placement=False,
                        device_count = {'CPU': 1})
session = tf.Session(config=config)
keras.backend.set_session(session)


class Textonator(BaseEstimator):
    """Textator"""

    def __init__(self,

                 w2v_size      = 100,
                 w2v_window    = 5,
                 w2v_min_count = 5,
                 w2v_workers   = 16,
                 w2v_sg        = 0,
                 w2v_negative  = 5,

                 max_sequence_length      = 10,
                 wv_dim                   = 100,
                 pad_sequences_padding    = "pre",
                 pad_sequences_truncating = "post",

                 dropout1d        = 0.2,
                 lstm_w           = 64,
                 dense_dropout    = 0.2,
                 dense_activation = 'sigmoid',

                 compile_loss      = 'categorical_crossentropy',
                 compile_optimazer = Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99),
                 compile_metrics   = ["acc"],

                 validation_split = 0.2,
                 epochs           = 10,
                 batch_size       = 256,

                 best_params = None
                ):
        """
        Called when initializing the classifier
        """
        # setup huperparams
        self.dropout1d           = dropout1d
        self.lstm_w              = lstm_w
        self.dense_dropout       = dense_dropout
        self.dense_activation    = dense_activation

        self.compile_loss        = 'binary_crossentropy'
        self.compile_optimazer   = Adam(lr=0.001, clipnorm=.25, beta_1=0.7, beta_2=0.99)
        self.compile_metrics     = ["acc"]

        self.classes             = classes,

        self.w2v_size            = w2v_size,
        self.w2v_window          = w2v_window,
        self.w2v_min_count       = w2v_min_count,
        self.w2v_workers         = w2v_workers,
        self.w2v_sg              = w2v_sg,
        self.w2v_negative        = w2v_negative

        self.classes             = classes

        self.wv_dim              = wv_dim
        self.max_sequence_length = max_sequence_length

        self.validation_split    = validation_split
        self.epochs              = epochs
        self.batch_size          = batch_size

        self.pad_sequences_padding    = pad_sequences_padding
        self.pad_sequences_truncating = pad_sequences_truncating

        # setup tools
        self.vocab      = Counter()
        self.vocab_1    = Counter()
        self.vocab_2    = Counter()
        self.tokenizer  = WordPunctTokenizer()
        self.morph      = pymorphy2.MorphAnalyzer()
        self.white_list = string.ascii_lowercase + "ёйцукенгшщзхъїфыівапролджэєячсмитьбю`'"
        self.stop_words =  set(
            [x for x in self.white_list]
            + stopwords.words('russian')
            + get_stop_words('en')
            + get_stop_words('ru')
            + get_stop_words('uk')
            + stopwords.words('english')
            + [
                'для',
                'весь',
                'по',
                'как',
                'на',
                'под',
              ]
        )

        # TODO:
        # replace url
        #re_url = re.compile(r"((http|https)\:\/\/)?[a-zA-Z0-9\.\/\?\:@\-_=#]+\
        #                    .([a-zA-Z]){2,6}([a-zA-Z0-9\.\&\/\?\:@\-_=#])*",
        #                    re.MULTILINE|re.UNICODE)
        # replace ips
        #re_ip = re.compile("\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}")
        filtering = lambda x: x in self.white_list

        # set best params
        if best_params != None:
            for key in best_params.keys():
                self.__dict__[key] = best_params[key]

    def _data_filter(self, data, update_vocab=False):
        data_processed = []
        for wd in data:

            wd = wd.strip()
            wd = "".join(list(filter(filtering, wd)))
            if wd not in self.stop_words \
                    and wd not in string.punctuation \
                    and wd not in string.whitespace:

                lemmatized_word = self.morph.parse(wd)[0].normal_form
                if lemmatized_word:
                    data_processed.append(lemmatized_word)
        if update_vocab:
            self.vocab.update(data_processed)
        return data_processed

    def _text_to_wordlist(self, text, lower=False):
        # Tokenize
        text = self.tokenizer.tokenize(text)

        # optional: lower case
        if lower:
            text = [t.lower() for t in text]

        return text

    def _process_comments(self, list_sentences, update_vocab=True, lower=True):
        comments = []
        for text in tqdm(list_sentences):
            txt = self._text_to_wordlist(text, lower=lower)
            txt = self._data_filter(txt, update_vocab=update_vocab)
            comments.append(txt)
        return comments

    def _fit_w2v(self, X, y=None, verbose=0):
        # word normalization
        doc = self._process_comments(X)

        # Word2Vec
        w2v_model = Word2Vec(doc, size=self.w2v_size, window=self.w2v_window, min_count=self.w2v_min_count, workers=self.w2v_workers, sg=self.w2v_sg, negative=self.w2v_negative)

        if verbose:
            print("Number of word vectors: {}".format(len(w2v_model.wv.vocab)))

        self.w2v_model = w2v_model
        return w2v_model,

    def _create_model(self, wv_matrix_1, wv_matrix_2, nb_words_1, nb_words_2):
        # Embedding
        wv_layer_1 = Embedding(
                     nb_words_1,
                     self.wv_dim,
                     mask_zero=False,
                     weights=[wv_matrix_1],
                     input_length=self.max_sequence_length,
                     trainable=False)

        wv_layer_2 = Embedding(
                     nb_words_2,
                     self.wv_dim,
                     mask_zero=False,
                     weights=[wv_matrix_2],
                     input_length=self.max_sequence_length,
                     trainable=False)

        # Inputs 1
        comment_input_1 = Input(shape=(self.max_sequence_length,), dtype='int32', name="X1")
        embedded_sequences_1 = wv_layer_1(comment_input_1)

        # Inputs 2
        comment_input_2 = Input(shape=(self.max_sequence_length,), dtype='int32', name="X2")
        embedded_sequences_2 = wv_layer_2(comment_input_2)

        # concatenate
        embedded_sequences = concatenate([(embedded_sequences_1), (embedded_sequences_2)])

        # biLSTm
        embedded_sequences = SpatialDropout1D(self.dropout1d)(embedded_sequences)
        x = Bidirectional(LSTM(self.lstm_w, return_sequences=False))(embedded_sequences)

        # Output
        x = Dropout(self.dense_dropout)(x)
        x = BatchNormalization()(x)
        preds = Dense(1, activation=self.dense_activation)(x)

        # build the model
        model = Model(inputs=[comment_input_1, comment_input_2], outputs=preds)

        # compile model
        model.compile(loss      = self.compile_loss,
                      optimizer = self.compile_optimazer,
                      metrics   = self.compile_metrics)

        return model,

    def _create_wv_matrix(self, w2v_model, word_index, nb_words, max_nb_words):
        wv_matrix = (np.random.rand(nb_words, self.wv_dim) - 0.5) / 5.0
        for word, i in word_index.items():
            if i >= max_nb_words:
                continue
            try:
                embedding_vector = w2v_model.wv[word]
                wv_matrix[i]     = embedding_vector
            except:
                # TODO something
                pass
        return wv_matrix

    def _normalize(self, X, word_index, update_vocab=True):
        doc = self._process_comments(X, update_vocab=update_vocab)
        sequences = [[word_index.get(t, 0) for t in comment] for comment in doc]
        data = pad_sequences(sequences,
                             maxlen    = self.max_sequence_length,
                             padding   = self.pad_sequences_padding,
                             truncating = self.pad_sequences_truncating)
        return data

    def fit(self, X1, X2, y, w2v_model_1=None, w2v_model_2=None, verbose=0):
        """
            This should fit classifier. All the "work" should be done here.
        """
        # train or load 1
        if w2v_model_1 == None:
            w2v_model_1  = self._fit_w2v(X1, y, verbose)
            self.vocab_1 = self.vocab
            self.vocab   = Counter()
        elif type(w2v_model_1) == str:
            w2v_model_1 = Word2Vec.load(w2v_model_1)
        else:
            w2v_model_1 = w2v_model_1

        # train or load 2
        if w2v_model == None:
            w2v_model_2  = self._fit_w2v(X2, y, verbose)
            self.vocab_2 = self.vocab
            self.vocab   = Counter()
        elif type(w2v_model) == str:
            w2v_model_2 = Word2Vec.load(w2v_model_2)
        else:
            w2v_model_2 = w2v_model_2

        max_nb_words_1 = len(w2v_model_1.wv.vocab)
        word_index_1 = {t[0]: i+1 for i,t in enumerate(self.vocab_1.most_common(max_nb_words_1))}
        nb_words_1 = min(max_nb_words_1, len(w2v_model_1.wv.vocab)) + 1

        max_nb_words_2 = len(w2v_model_2.wv.vocab)
        word_index_2 = {t[0]: i+1 for i,t in enumerate(self.vocab_2.most_common(max_nb_words_2))}
        nb_words_2 = min(max_nb_words_2, len(w2v_model_2.wv.vocab)) + 1

        # normalize
        X_train_1 = self._normalize(X1, word_index_1, update_vocab=False)
        X_train_2 = self._normalize(X2, word_index_2, update_vocab=False)
        y_train = y
        if verbose:
            print('Len of X1 tensor:', len(X_train_1))
            print('Len of X2 tensor:', len(X_train_2))
            print('Len of y tensor:', len(y_train))

        wv_matrix_1 = self._create_wv_matrix(w2v_model_1, word_index_1, nb_words_1, max_nb_words_1)
        wv_matrix_2 = self._create_wv_matrix(w2v_model_2, word_index_2, nb_words_2, max_nb_words_2)

        model = self._create_model(wv_matrix_1, wv_matrix_2, nb_words_1, nb_words_2)

        # train
        hist = model.fit([X_train], y_train, validation_split=validation_split,
                 epochs=epochs, batch_size=batch_size, shuffle=True)

        self.model        = model
        self.word_index_1 = word_index_1
        self.word_index_1 = word_index_2
        return hist


    def evaluate(self, X1, X2, y, verbose=0):
        """
        Test
        """
        self.vocab = self.vocab_1
        X_test_1 = self._normalize(X, self.word_index_1, update_vocab=False)
        self.vocab = self.vocab_2
        X_test_2 = self._normalize(X, self.word_index_2, update_vocab=False)
        self.vocab = Counter()

        y_test = y
        if verbose:
            print('Len of X1 tensor:', len(X_test_1))
            print('Len of X2 tensor:', len(X_test_2))
            print('Len of y tensor:', len(y_test))

        return self.model.evaluate({"X1": [X_test_1], "X2": [X_test_2]}, y_test)

    def predict(self,  X1, X2, y=None):
        """
        Predict
        """
        # normalize
        self.vocab = self.vocab_1
        X_test_1 = self._normalize(X, self.word_index_1, update_vocab=False)
        self.vocab = self.vocab_2
        X_test_2 = self._normalize(X, self.word_index_2, update_vocab=False)
        self.vocab = Counter()

        # predict
        return self.model.predict({"X1": [X_test_1], "X2": [X_test_2]})