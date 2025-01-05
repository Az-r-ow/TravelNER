from abc import ABC, abstractmethod
import os
import tensorflow as tf
import numpy as np
from transformers import TFCamembertForTokenClassification, CamembertTokenizerFast
import pandas as pd
from .data_processing import (
    process_sentence,
    encode_and_pad_sentence,
    encode_and_pad_sentence_pos,
)
from .metrics import masked_loss, masked_accuracy, entity_accuracy
import stanza

nlp = stanza.Pipeline("fr", processors="tokenize,pos")


class NERModel(ABC):
    file_path = os.path.dirname(os.path.abspath(__file__))
    vocab_path = os.path.join(file_path, "vocab.pkl")
    pos_tags_path = os.path.join(file_path, "pos_tags.pkl")
    vocab = pd.read_pickle(vocab_path)
    pos_tags = pd.read_pickle(pos_tags_path)

    @abstractmethod
    def get_entities(self, text: str):
        pass

    @abstractmethod
    def predict(self, text: str):
        pass

    @abstractmethod
    def encode_sentence(self, sentence: str):
        pass


class LSTM_NER(NERModel):
    def __init__(self):
        self.model_path = os.path.join(
            self.file_path, "models", "lstm_with_pos", "model.keras"
        )
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                "masked_loss": masked_loss,
                "masked_accuracy": masked_accuracy,
                "entity_accuracy": entity_accuracy,
                "log_softmax_v2": tf.nn.log_softmax,
            },
        )

    def encode_sentence(self, sentence: str):
        processed_sentence = process_sentence(
            sentence, stemming=True, return_tokens=True
        )
        encoded_sentence = tf.convert_to_tensor(
            [encode_and_pad_sentence(processed_sentence, self.vocab, max_length=100)]
        )
        sentence_pos = nlp(sentence)
        pos_tags = [word.upos for sent in sentence_pos.sentences for word in sent.words]
        encoded_pos = tf.convert_to_tensor(
            [encode_and_pad_sentence_pos(pos_tags, self.pos_tags, max_length=100)]
        )
        return [encoded_sentence, encoded_pos]

    def get_entities(self, text: str):
        encoded_sentence = self.encode_sentence(text)
        predictions = self.predict(encoded_sentence)
        return predictions[0].numpy()

    def predict(self, encoded_sentence):
        return tf.math.argmax(self.model.predict(encoded_sentence, verbose=0), axis=-1)


class BiLSTM_NER(NERModel):
    def __init__(self):
        self.model_path = os.path.join(
            self.file_path, "models", "bilstm", "model.keras"
        )
        self.model = tf.keras.models.load_model(
            self.model_path,
            custom_objects={
                "masked_loss": masked_loss,
                "masked_accuracy": masked_accuracy,
                "entity_accuracy": entity_accuracy,
                "log_softmax_v2": tf.nn.log_softmax,
            },
        )

    def encode_sentence(self, sentence: str):
        processed_sentence = process_sentence(
            sentence, stemming=True, return_tokens=True
        )
        encoded_sentence = tf.convert_to_tensor(
            [encode_and_pad_sentence(processed_sentence, self.vocab, max_length=100)]
        )
        return encoded_sentence

    def get_entities(self, text: str):
        encoded_sentence = self.encode_sentence(text)
        predictions = self.predict(encoded_sentence)
        return predictions[0].numpy()

    def predict(self, encoded_sentence):
        return tf.math.argmax(self.model.predict(encoded_sentence, verbose=0), axis=-1)


class CamemBERT_NER(NERModel):
    def __init__(self, num_labels=3):
        self.model = TFCamembertForTokenClassification.from_pretrained(
            "Az-r-ow/CamemBERT-NER-Travel", num_labels=num_labels
        )

        self.tokenizer = CamembertTokenizerFast.from_pretrained(
            "cmarkea/distilcamembert-base"
        )

    def encode_sentence(self, sentence: str):
        return self.tokenizer(
            sentence,
            return_tensors="tf",
            padding="max_length",
            max_length=150,
        )

    def get_entities(self, text: str):
        encoded_sentence = self.encode_sentence(text)
        predictions = self.predict(encoded_sentence).logits
        predictions = tf.math.argmax(predictions, axis=-1)[0].numpy()
        return self.align_labels_with_original_sentence(
            encoded_sentence, [predictions]
        )[0]

    def predict(self, encoded_sentence):
        return self.model.predict(encoded_sentence)

    def align_labels_with_original_sentence(self, tokenized_inputs, predictions):
        """
        Aligns predictions from token classification back to the original sentence words.

        Args:
            tokenized_inputs (BatchEncoding): Tokenized input from CamembertTokenizerFast.
            predictions (np.array): Model predictions, shape (batch_size, seq_len, num_labels).

        Returns:
            List[List[str]]: Adjusted labels for each word in the original sentences.
        """
        aligned_labels = []

        for i in range(len(predictions)):  # Iterate through each example in the batch
            word_ids = tokenized_inputs.word_ids(
                batch_index=i
            )  # Get word IDs for this example
            sentence_labels = []
            current_word = None
            word_label = None

            for token_idx, word_idx in enumerate(word_ids):
                # Skip special tokens where word_idx is None
                if word_idx is None:
                    continue

                # If we're at a new word
                if word_idx != current_word:
                    # Append label for the completed word
                    if current_word is not None:
                        sentence_labels.append(word_label)

                    print(i)
                    print(token_idx)
                    # Reset for the new word
                    current_word = word_idx
                    word_label = predictions[i][token_idx]

                # Handle subwords (optional: take the first or last subword label)
                else:
                    # Here, we take the first subword label; alternatively, update word_label as needed.
                    continue

            # Append the last word's label
            if current_word is not None:
                sentence_labels.append(word_label)

            aligned_labels.append(sentence_labels)

        return aligned_labels
