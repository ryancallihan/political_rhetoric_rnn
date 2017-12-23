import os
import numpy as np
import data_processing as dp
from train import train
from predict import tweet_predict, word_predict
from config import Config


def run():
    sections = ["2015_1q", "2015_2q", "2015_3q", "2015_4q", "2016_1q", "2016_2q", "2016_3q", "2016_4q", "2017_1q",
                "2017_1q_r", "2017_2q", "2017_2q_r"]

    w2v_path = 'data/twitter2vec.w2v'
    tokenizers = {"x": "data/x_tokenizer.p", "y": "data/y_tokenizer.p"}

    processing = dp.Data_Processing(tokenizers)
    config = Config()

    for quarter in sections:
        x_train, y_train, x_test, y_test, x_tokenizer, y_tokenizer = processing.run(
            data_path=os.path.join("data", ''.join([quarter, "_data.tsv"])),
            x_mode="index",
            y_mode="vectorize")

        num_words = len(x_tokenizer.word_index) + 1
        num_classes = len(y_tokenizer.word_index.items())

        w2v_layer = processing.create_embedding_layer(w2v_path)

        train(data_path=quarter,
              w2v_layer=w2v_layer,
              config=config,
              num_words=num_words,
              num_classes=num_classes,
              data={"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test},
              from_saved=False)

        # Chooses most indicative tweets for each quarter
        tweet_predict(data_path=quarter,
                      data={"x_test": x_test, "y_test": y_test},
                      processing=processing,
                      config=config,
                      w2v_layer=w2v_layer)

        # Chooses most indicative words for each quarter
        word_predict(data_path=quarter,
                     processing=processing,
                     w2v_layer=w2v_layer,
                     config=config)


if __name__ == "__main__":
    run()
