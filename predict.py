import os
import numpy as np
import tensorflow as tf
from tensorflow.python.framework import ops
import data_processing as dp
from model import Model, Phase
from keras.preprocessing.text import Tokenizer


def predict(config, validation_batches, word_vector, save_path, texts=None, pred_type="tweet"):
    """
    Predicts scores of inputs
    :param config:
    :param validation_batches:
    :param word_vector:
    :param save_path:
    :param texts:
    :param pred_type:
    :return:
    """

    validation_batches, validation_lens, validation_labels, n_word = validation_batches

    with open("".join(["tf_prediction_results/tweet_prediction_", save_path, "_", pred_type, ".tsv"]), "w",
              encoding="utf-8") \
            as history:

        history.write("text\t" +
                      "hp_label\t" +
                      "pred_label\t" +
                      "D_prob\t" +
                      "R_prob\t" +
                      "time\t")

        history.close()

    # Resets graph
    ops.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:

        with tf.variable_scope(save_path, reuse=None):
            validation_model = Model(
                config=config,
                batch=validation_batches,
                label_batch=validation_labels,
                word_vector=word_vector,
                phase=Phase.Validation)

        # Initialize Saver
        saver = tf.train.Saver()

        print("LOADING FROM SAVED MODEL>>>>")
        init_op = tf.global_variables_initializer()
        sess.run(init_op)
        saver.restore(sess, os.path.join("tf_saved_models", save_path, save_path))
        print(save_path, "Model restored.")

        # Train on all batches.

        tweet_idx = 0
        with open("".join(["tf_prediction_results/tweet_prediction_", save_path, "_", pred_type, ".tsv"]), "a",
                  encoding="utf-8") as history:
            accuracy = 0.0
            for batch in range(validation_batches.shape[0]):
                loss, acc, y_true, y_pred, logits = sess.run(
                    [validation_model.loss, validation_model.accuracy, validation_model.hp_labels,
                     validation_model.labels, validation_model.logits], {
                        validation_model.x: validation_batches[batch], validation_model.y: validation_labels[batch]})
                accuracy += acc

                for t_idx in range(config.BATCH_SIZE):
                    # Binary
                    history.write("\n%s\t%d\t%d\t%.10f\t%.10f\t%s" %
                                  (texts[tweet_idx], y_true[t_idx], y_pred[t_idx], logits[t_idx][0],
                                   logits[t_idx][1], save_path))

                    tweet_idx += 1

        history.close()

    sess.close()
    del sess, validation_model
    print_top10(save_path, "".join(["tf_prediction_results/tweet_prediction_", save_path, "_", pred_type, ".tsv"]))


def tweet_predict(data_path, data, config, processing, w2v_layer):
    """
    Creates TSV with scores of all tweets
    :param data_path:
    :param data:
    :param config:
    :param processing:
    :param w2v_layer:
    :return:
    """
    print("PREDICTING TWEETS FOR SECTION:", data_path)

    tweets_train, _, tweets_test, _, x_tokenizer, y_tokenizer = processing.run(
        data_path=os.path.join("data", ''.join([data_path, "_data.tsv"])),
        shuffle=False)

    num_words = len(x_tokenizer.word_index) + 1
    num_classes = len(y_tokenizer.word_index.items())

    print("NUM WORDS: ", num_words, " NUM CLASSES: ", num_classes)
    print(data["x"].shape)
    print(data["y"].shape)
    validation_batches = dp.generate_instances(
        data["x"],
        data["y"],
        num_words,
        num_classes,
        config.MAX_TIMESTEPS,
        batch_size=config.BATCH_SIZE)

    # Train the model
    predict(config, validation_batches, w2v_layer, data_path, np.append(tweets_train, tweets_test))
    print("FINISHED:", data_path, "TWEET PREDICTION\n\n")


def word_predict(data_path, processing, w2v_layer, config):
    """
    Creates TSV with scores of all vocabulary.
    :param data_path:
    :param processing:
    :param w2v_layer:
    :param config:
    :return:
    """
    print("PREDICTING WORD FOR SECTION:", data_path)

    data1, _, data2, _, x_tokenizer, y_tokenizer = processing.run(
        data_path=os.path.join("data", ''.join([data_path, "_data.tsv"])),
        y_mode="vectorize")

    get_counts = Tokenizer()
    get_counts.fit_on_texts(np.concatenate((data1, data2)))

    words = [word[0] for word in get_counts.word_index.items()]

    word_vectors = dp.pad_sequences(processing.index_text(words), max_len=30)

    labels = [[0, 1] for label in range(len(words))]

    num_words = len(x_tokenizer.word_index) + 1
    num_classes = len(y_tokenizer.word_index.items())

    validation_batches = dp.generate_instances(
        word_vectors,
        labels,
        num_words,
        num_classes,
        config.MAX_TIMESTEPS,
        batch_size=config.BATCH_SIZE)

    # Train the model
    predict(config, validation_batches, w2v_layer, data_path, words, pred_type="words")
    print("FINISHED:", data_path, "WORD PREDICTION\n\n")


def print_top10(quarter, tsv_path):
    predictions = dp.load_tsv(tsv_path)
    predictions = predictions[(predictions["time"] == quarter)]
    dems_idx = np.argsort(predictions["D_prob"])[::-1]
    rep_idx = np.argsort(predictions["R_prob"])[::-1]
    print("TOP 10 MOST INDICATIVE TEXT FOR", quarter)
    print("\nREPUBLICANS>>>")
    _ = [print("\t", (i+1), "-", text) for i, text in enumerate(np.array(predictions["text"])[rep_idx[:10]])]
    print("\nDEMOCRATS>>>")
    _ = [print("\t", (i+1), "-", text) for i, text in enumerate(np.array(predictions["text"])[dems_idx[:10]])]
