import gensim
import numpy as np
import pandas as pd
import pickle as p
from keras.preprocessing.text import Tokenizer
from sklearn.model_selection import train_test_split

def generate_instances(
        data,
        labels_data,
        n_word,
        n_label,
        max_timesteps,
        batch_size=128):
    """
    prepares batches for tensorflow model.
    :param data:
    :param labels_data:
    :param n_word:
    :param n_label:
    :param max_timesteps:
    :param batch_size:
    :return:
    """
    n_batches = len(data) // batch_size

    # We are discarding the last batch for now, for simplicity.
    labels = np.zeros(
        shape=(
            n_batches,
            batch_size,
            n_label),
        dtype=np.float32)
    lengths = np.zeros(
        shape=(
            n_batches,
            batch_size),
        dtype=np.int32)
    words = np.zeros(
        shape=(
            n_batches,
            batch_size,
            max_timesteps),
        dtype=np.int32)

    for batch in range(n_batches):
        for idx in range(batch_size):
            word = data[(batch * batch_size) + idx]
            # Add label distribution

            label = labels_data[(batch * batch_size) + idx]
            index = np.nonzero(label)[0][0]
            labels[batch, idx, index] = 1

            # Sequence
            timesteps = min(max_timesteps, len(word))

            # Sequence length (time steps)
            lengths[batch, idx] = timesteps

            # Word characters
            words[batch, idx, :timesteps] = word[:timesteps]

    return words, lengths, labels, n_word


def load_tsv(filename):
    """
    Loads pandas dataframe from file
    :param filename:
    :return:
    """
    return pd.read_csv(filename, sep='\t', error_bad_lines=False)


def load_dataframe(filename):
    """
    Returns pandas dataframe from pickle
    :param filename:
    :return:
    """
    print("FILENAME: ", filename)
    return p.load(open(filename, "rb"))


def shuffle_dataframe(dataframe):
    """
    Shuffles rows of pandas dataframe
    :param dataframe:
    :return:
    """
    return dataframe.iloc[np.random.permutation(len(dataframe))]


def retrieve_features(dataframe):
    """
    Retrieves features (X) from dataframe
    :param dataframe:
    :return:
    """
    return list(dataframe["tweet"])


def retrieve_labels(dataframe):
    """
    Retrieves labels (Y) from pandas dataframe
    Returns years | months | partys
    :param dataframe:
    :return:
    """
    years = list([str(int(year)) for year in dataframe["year"]])
    months = list([int(month) for month in dataframe["month"]])
    partys = list(dataframe["affiliation"])
    for idx, month in enumerate(months):
        month = int(month)
        if 0 < month <= 4:
            months[idx] = str(1)
        elif 3 < month <= 6:
            months[idx] = str(2)
        elif 6 < month <= 9:
            months[idx] = str(3)
        elif 9 < month <= 12:
            months[idx] = str(4)
        else:
            months[idx] = str("XXX")

    return years, months, partys


def load_w2v_model(w2v_path):
    """
    Loads pretrained w2v model
    :param w2v_path:
    :return:
    """
    return gensim.models.Word2Vec.load(w2v_path)


def pad_sequences(sentences, max_len):
    """
    Pads the end of a sentence or cuts it short.
    :param sentences:
    :param max_len:
    :return:
    """
    padded_sentences = []
    for sent in sentences:
        padded_sent = []
        sent_len = len(sent)
        for idx in range(1, max_len):
            if idx <= sent_len:
                padded_sent.append(sent[idx - 1])
            else:
                padded_sent.append(0)
        padded_sentences.append(padded_sent)
    return padded_sentences


class Data_Processing:
    def __init__(self, tokenizers):
        self.x_tokenizer = p.load(file=open(tokenizers["x"], "rb"))
        self.y_tokenizer = p.load(file=open(tokenizers["y"], "rb"))

    def vectorize_text(self, texts, is_label=False):
        """
        Takes a list of texts and returns a matrix of "bag of word" vectors.
        Each row represents a text with a vector of length: number of types.
        includes a 1 for each type in the text.
        :param is_label:
        :param texts:
        :return:
        """
        if is_label:
            tokenizer = self.y_tokenizer
        else:
            tokenizer = self.x_tokenizer
        indexed = tokenizer.texts_to_sequences(texts)
        num_items = len(tokenizer.word_index)
        vectors = []
        for index in indexed:
            index = index
            vec = np.zeros(num_items, dtype=np.int).tolist()
            for idx in index:
                vec[idx - 1] = 1
            vectors.append(vec)
        return vectors

    def index_to_text(self, sentence, is_label=False):
        """
        Converts a sentence from indices to words
        :param sentence:
        :param is_label:
        :return:
        """
        if is_label:
            tokenizer = self.y_tokenizer
        else:
            tokenizer = self.x_tokenizer
        word_map = dict(map(reversed, tokenizer.word_index.items()))
        converted_sentence = []
        for word in sentence:
            if word != 0 or word != "0":
                converted_sentence.append(word_map[word])
        return converted_sentence

    def create_embedding_layer(self, w2v_path, is_label=False):
        """
        Creates an embeddings layer for Tensorflow model.
        :param w2v_path:
        :param is_label:
        :return:
        """
        if is_label:
            tokenizer = self.y_tokenizer
        else:
            tokenizer = self.x_tokenizer
        w2v_model = load_w2v_model(w2v_path)
        num_words = len(tokenizer.word_index) + 1
        w2v_size = w2v_model.vector_size
        w2v_embeddings = np.zeros((num_words, w2v_size))
        for word, index in tokenizer.word_index.items():
            try:
                w2v_embeddings[index] = w2v_model[word]
            except KeyError:
                continue
        return w2v_embeddings

    def build_vectors(self, train, test, is_label=False):
        """
        Builds vectors of indices
        :param train:
        :param test:
        :param is_label:
        :return:
        """
        if is_label:
            tokenizer = self.y_tokenizer
        else:
            tokenizer = self.x_tokenizer
        train_vectorized = self.vectorize_text(train, tokenizer)
        test_vectorized = self.vectorize_text(test, tokenizer)
        return train_vectorized, test_vectorized

    def index_text(self, texts, is_label=False):
        """
        Indexes text
        :param texts:
        :param is_label:
        :return:
        """
        if is_label:
            tokenizer = self.y_tokenizer
        else:
            tokenizer = self.x_tokenizer
        return tokenizer.texts_to_sequences(texts)

    def build_indices(self, train, test, maxlen=-1, pad=True, is_label=False):
        """
        Creates index vectors
        :param train:
        :param test:
        :param maxlen:
        :param pad:
        :param is_label:
        :return:
        """
        if maxlen == -1:
            maxlen = len(
                max(train, key=len)
            )

        train = self.index_text(train, is_label=is_label)
        test = self.index_text(test, is_label=is_label)

        if pad:
            train = pad_sequences(train, max_len=maxlen)
            test = pad_sequences(test, max_len=maxlen)
        return train, test

    def run(self, data_path, x_mode=None, y_mode=None,
            feat_vec_len=-1, shuffle=True, test_size=0.2):
        """
        Returns all training and testing data
        (x_train, y_train, x_test, y_test, x_tokenizer, y_tokenizer)
        :param data_path:
        :param test_size:
        :param shuffle:
        :param feat_vec_len:
        :param y_mode:
        :param x_mode:
        :return:
        """

        data = load_tsv(data_path)

        print("DATA LEN: ", len(data))
        x = retrieve_features(data)
        y_years, y_months, y_parties = retrieve_labels(data)

        y = y_parties

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, shuffle=shuffle)

        # Prepare feature data structure
        if x_mode == "index":
            x_train, x_test = self.build_indices(x_train, x_test, maxlen=feat_vec_len)

        if y_mode == "vectorize":
            y_train, y_test = self.build_vectors(y_train, y_test, is_label=True)

        return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test), self.x_tokenizer, self.y_tokenizer
