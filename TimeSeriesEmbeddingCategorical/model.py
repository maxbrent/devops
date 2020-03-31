import tensorflow as tf
from tensorflow.keras.layers import Activation


class EmbeddingModel(tf.keras.models.Model):
    def __init__(self, hidden_units=3, output_units=1, embeddings_initializer=tf.random.normal,
                 kernel_initializer=tf.keras.initializers.he_uniform(seed=0), dropout_rate=0.4, activation="sigmoid",
                 trainable=True,
                 **kwargs):
        super().__init__(**kwargs)

        vocab = [min(50, (1220 + 1) / 2), min(50, (1519 + 1) / 2), min(50, (222 + 1) / 2), min(50, (222 + 1) / 2),
                 min(50, (2218 + 1) / 2)]

        self.embeded2 = tf.keras.layers.Embedding(1220, vocab[0], trainable=trainable, embeddings_initializer=embeddings_initializer)
        self.embeded3 = tf.keras.layers.Embedding(1519, vocab[1], trainable=trainable, embeddings_initializer=embeddings_initializer)
        self.embeded4 = tf.keras.layers.Embedding(222, vocab[2], trainable=trainable, embeddings_initializer=embeddings_initializer)
        self.embeded5 = tf.keras.layers.Embedding(222, vocab[3], trainable=trainable, embeddings_initializer=embeddings_initializer)
        self.embeded6 = tf.keras.layers.Embedding(2218, vocab[4], trainable=trainable, embeddings_initializer=embeddings_initializer)

        self.flattened = tf.keras.layers.Flatten()
        self.hidden1 = tf.keras.layers.Dense(hidden_units, kernel_initializer=kernel_initializer)
        self.normalize1 = tf.keras.layers.BatchNormalization()
        self.regularize1 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.activator_1 = Activation("relu")
        self.hidden2 = tf.keras.layers.Dense(hidden_units, kernel_initializer=kernel_initializer)
        self.normalize_2 = tf.keras.layers.BatchNormalization()
        self.regularize2 = tf.keras.layers.Dropout(rate=dropout_rate)
        self.activator_2 = Activation("relu")
        self.out = tf.keras.layers.Dense(output_units, activation=activation)

    def call(self, vals):
        input2, input3, input4, input5, input6 = vals

        embeded2 = self.embeded2(input2)
        embeded3 = self.embeded2(input3)
        embeded4 = self.embeded2(input4)
        embeded5 = self.embeded2(input5)
        embeded6 = self.embeded2(input6)

        merged_embeddings = tf.keras.layers.concatenate([embeded2, embeded3, embeded4, embeded5, embeded6])
        flattened = self.flattened(merged_embeddings)
        hidden1 = self.hidden1(flattened)
        normalize_1 = self.normalize1(hidden1)
        regularize1 = self.regularize1(normalize_1)
        activator_1 = self.activator_1(regularize1)
        hidden2 = self.hidden2(activator_1)
        normalize_2 = self.normalize_2(hidden2)
        regularize2 = self.regularize2(normalize_2)
        activator_2 = self.activator_2(regularize2)

        output = self.out(activator_2)

        return output
