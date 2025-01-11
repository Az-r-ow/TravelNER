import tensorflow as tf


class LSTM:
    def __init__(self, vocab, nb_labels: int, pos_tags: list, emb_dim=100, emb_size=32):
        word_input = tf.keras.layers.Input(shape=(emb_dim,), name="word_input")
        pos_input = tf.keras.layers.Input(shape=(emb_dim,), name="pos_input")

        word_embedding = tf.keras.layers.Embedding(
            len(vocab), emb_size, name="word_embedding"
        )(word_input)

        pos_embedding = tf.keras.layers.Embedding(
            len(pos_tags),
            emb_size,
            name="pos_embedding",
        )(pos_input)

        concatenated = tf.keras.layers.Concatenate()([word_embedding, pos_embedding])

        masked_cat = tf.keras.layers.Masking(mask_value=0)(concatenated)

        lstm_layer_with_pos = tf.keras.layers.LSTM(
            emb_size, return_sequences=True, name="lstm_layer"
        )(masked_cat)

        dropout = tf.keras.layers.Dropout(0.2)(lstm_layer_with_pos)

        output = tf.keras.layers.Dense(nb_labels, activation=tf.nn.log_softmax)(dropout)

        self.model = tf.keras.Model(inputs=[word_input, pos_input], outputs=output)

    def load_from_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def predict(self, x, verbose=0):
        return self.model.predict(x, verbose=verbose)
