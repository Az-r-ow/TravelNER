from app.travel_resolver.libs.nlp.ner.models import LSTM_NER, BiLSTM_NER, CamemBERT_NER
import tensorflow as tf

print(tf.__version__)

ner_model = LSTM_NER()

sentence = "Je voudrais voyager de Nice à Clermont Ferrand."

print(ner_model.get_entities(sentence))
