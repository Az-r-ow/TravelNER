import tensorflow as tf


class BiLSTM:
    def __init__(self, vocab, nb_labels, emb_dim=100):
        self.model = tf.keras.models.Sequential(
            layers=[
                tf.keras.layers.Embedding(len(vocab) + 1, emb_dim, mask_zero=True),
                tf.keras.layers.Bidirectional(
                    tf.keras.layers.LSTM(emb_dim, return_sequences=True)
                ),
                tf.keras.layers.Dropout(0.3),
                tf.keras.layers.Dense(nb_labels, activation=tf.nn.log_softmax),
            ]
        )

    def load_from_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, x, verbose=0):
        return self.model.predict(x, verbose=verbose)
