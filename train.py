import os
import timeit

import tensorflow as tf
from sklearn.metrics import f1_score, recall_score, precision_score
from config import Config
from tensorflow.python.framework import ops
import data_processing as dp
from model import Model, Phase
from keras.preprocessing.text import Tokenizer


def train_model(config, train_batches, validation_batches, word_vector, save_path, from_saved=False):
    """
    Trains RNN model
    :param config:
    :param train_batches:
    :param validation_batches:
    :param word_vector:
    :param save_path:
    :param from_saved:
    :return:
    """
    train_start_time = timeit.default_timer()
    train_batches, train_lens, train_labels, n_word = train_batches
    validation_batches, validation_lens, validation_labels, n_word = validation_batches

    if not from_saved and save_path == "2015_1q":
        with open("tf_training_hist/training_history.tsv", "w") as history:
            history.write("epoch\t" +
                          "train_loss\t" +
                          "val_loss\t" +
                          "val_acc\t" +
                          "f1score\t" +
                          "recall\t" +
                          "precision\t" +
                          "time\t")
            history.close()

    ops.reset_default_graph()
    sess = tf.InteractiveSession()

    with tf.Session() as sess:
        with tf.variable_scope(save_path, reuse=False):
            train_model = Model(
                config=config,
                batch=train_batches,
                label_batch=train_labels,
                word_vector=word_vector,
                phase=Phase.Train)

        with tf.variable_scope(save_path, reuse=True):
            validation_model = Model(
                config=config,
                batch=validation_batches,
                label_batch=validation_labels,
                word_vector=word_vector,
                phase=Phase.Validation)

        # Initialize Saver
        saver = tf.train.Saver()

        if from_saved:
            print("LOADING FROM SAVED MODEL>>>>")
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            saver.restore(sess, os.path.join("tf_saved_models", save_path, save_path))
            print("Model restored.")
        else:
            sess.run(tf.global_variables_initializer())

        for epoch in range(config.N_EPOCHS):
            epoch_start_time = timeit.default_timer()
            train_loss = 0.0
            validation_loss = 0.0
            accuracy = 0.0
            recall = 0.0
            precision = 0.0
            f_score = 0.0
            # Train on all batches.
            for batch in range(train_batches.shape[0]):
                loss, _ = sess.run([train_model.loss, train_model.train_op], {
                    train_model.x: train_batches[batch], train_model.y: train_labels[batch]})
                train_loss += loss
            # validation on all batches.
            for batch in range(validation_batches.shape[0]):
                loss, acc, y_true, y_pred = sess.run(
                    [validation_model.loss, validation_model.accuracy, validation_model.hp_labels,
                     validation_model.labels], {
                        validation_model.x: validation_batches[batch], validation_model.y: validation_labels[batch]})

                validation_loss += loss
                accuracy += acc
                recall += recall_score(y_true, y_pred, average='micro')
                precision += precision_score(y_true, y_pred, average='micro')
                f_score += f1_score(y_true, y_pred, average='micro')

            train_loss /= train_batches.shape[0]
            validation_loss /= validation_batches.shape[0]
            accuracy /= validation_batches.shape[0]
            recall /= validation_batches.shape[0]
            precision /= validation_batches.shape[0]
            f_score /= validation_batches.shape[0]

            print(
                "epoch %d - train loss: %.3f, validation loss: %.3f, validation acc: %.3f, f1 score: %.3f, recall: "
                "%.3f, precision: %.3f, time: %s" %
                (epoch + 1, train_loss, validation_loss, accuracy * 100, f_score * 100, recall, precision, save_path))
            with open("tf_training_hist/training_history.tsv", "a") as history:
                history.write("\n%d\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%.3f\t%s" %
                              (epoch + 1, train_loss, validation_loss, accuracy * 100,
                               f_score * 100, recall, precision, save_path))
            history.close()

            # Saves model
            save_to_disk = saver.save(sess=sess, save_path=os.path.join("tf_saved_models", save_path, save_path))
            print("MODEL SAVED>>>>")
            print("TIME ELAPSED FOR EPOCH: %.2f MINUTES" % ((timeit.default_timer() - epoch_start_time) / 60))
        sess.close()
    print("TIME ELAPSED FOR TRAINING: %.2f MINUTES" % ((timeit.default_timer() - train_start_time) / 60))


def train(data_path, data, num_words, num_classes, w2v_layer, config, from_saved=False):
    """
    Trains RNN model with given save_path. Can resume training from saved model.
    :param data_path:
    :param data:
    :param num_words:
    :param num_classes:
    :param w2v_layer:
    :param config:
    :param from_saved:
    :return:
    """
    # Generate batches
    train_batches = dp.generate_instances(
        data["x_train"],
        data["y_train"],
        num_words,
        num_classes,
        config.MAX_TIMESTEPS,
        batch_size=config.BATCH_SIZE)
    validation_batches = dp.generate_instances(
        data["x_test"],
        data["y_test"],
        num_words,
        num_classes,
        config.MAX_TIMESTEPS,
        batch_size=config.BATCH_SIZE)

    # Train the model
    train_model(config, train_batches, validation_batches, w2v_layer, data_path, from_saved=from_saved)
    print("FINISHED:", data_path, "TRAINING\n\n")


