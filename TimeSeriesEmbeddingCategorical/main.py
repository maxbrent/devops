import pandas as pd
import numpy as np
import tensorflow as tf

from model import EmbeddingModel
from utils import ETLDataPipeline


def main():
    obj_etl = ETLDataPipeline("data/train.csv", "data/test.csv")
    train, test = obj_etl.read_data()
    train = obj_etl.drop_cols(
        ['id', 'bin_0', 'bin_1', 'bin_2', 'bin_3', 'bin_4', 'nom_0', 'nom_1', 'nom_2', 'nom_3', 'nom_4', 'ord_0',
         'ord_1', 'ord_2', 'ord_3', 'ord_4', 'ord_5', 'day', 'month'])
    train = obj_etl.convert_dtypes(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'])
    train = obj_etl.encoder(['nom_5', 'nom_6', 'nom_7', 'nom_8', 'nom_9'])
    target = obj_etl.get_target('target')
    X_train, X_valid, y_train, y_valid = obj_etl.get_train_test(train, target)

    val1 = np.reshape(X_train['nom_5'].values, (-1, 1))
    val2 = np.reshape(X_train['nom_6'].values, (-1, 1))
    val3 = np.reshape(X_train['nom_7'].values, (-1, 1))
    val4 = np.reshape(X_train['nom_8'].values, (-1, 1))
    val5 = np.reshape(X_train['nom_9'].values, (-1, 1))
    val6 = np.reshape(y_train.values, (-1, 1))

    val11 = np.reshape(X_valid['nom_5'].values, (-1, 1))
    val22 = np.reshape(X_valid['nom_6'].values, (-1, 1))
    val33 = np.reshape(X_valid['nom_7'].values, (-1, 1))
    val44 = np.reshape(X_valid['nom_8'].values, (-1, 1))
    val55 = np.reshape(X_valid['nom_9'].values, (-1, 1))
    val66 = np.reshape(y_valid.values, (-1, 1))

    tf.random.set_seed(0)

    # 100 is number of epochs, 32 is batch size
    s = 100 * len(X_train) // 32
    learning_rate = tf.keras.optimizers.schedules.ExponentialDecay(0.01, s, 0.1)
    opt = tf.keras.optimizers.Adam(learning_rate)

    model = EmbeddingModel(hidden_units=3, output_units=1, embeddings_initializer=tf.random.normal,
                           kernel_initializer=tf.keras.initializers.he_uniform(seed=0), dropout_rate=0.4,
                           activation="sigmoid",
                           trainable=True)
    model.compile(loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'], optimizer=opt)
    baseline_history = model.fit(
        (val1, val2, val3, val4, val5), val6, epochs=10,
        batch_size=32, validation_data=(
            (val11, val22, val33, val44, val55), val66),
        class_weight={0: 0.5, 1: 0.5})


if __name__ == "__main__":
    main()
