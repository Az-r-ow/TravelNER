import nltk, re
import tensorflow as tf
from tqdm import tqdm

# Will download the necessary resources for nltk
# Should skip if resources found
nltk.download("punkt_tab")


def get_tagged_content(sentence: str, tag: str) -> str | None:
    """
    Extract the content between two tags in a sentence given the tag.

    Args:
      sentence (str): The sentence to extract the content from.
      tag (str): The tag to extract the content between.

    Returns:
      str | None: The content between the tags. None if not found

    Raises:
      ValueError: If tag is not provided or tag not str.

    Example:
      >>> get_tagged_content("Je voudrais voyager de <Dep>Nice<Dep> à <Arr>Clermont Ferrand<Arr>.", "<Dep>")
      "Nice"
    """
    if not tag or not isinstance(tag, str):
        raise ValueError("tag must be a non-empty string")

    tag_match = re.search(rf"{tag}(.*?){tag}", sentence)
    if tag_match:
        return tag_match.group(1)
    return None


def process_sentence(
    sentence: str,
    stemming: bool = False,
    return_tokens: bool = False,
) -> str:
    """
    Given a sentence, apply some processing techniques to the sentence and return the processed sentence

     **Note**: We are stemming the tokens instead of lemmatizing them because stemming is faster and in our case
     we are interested in getting a response the fastest way possible.

     Args:
       sentence (str): The sentence to process.
       stemming (bool): Whether to stem the tokens.

     Returns:
       str: The processed sentence
    """
    tokenized_sentence = nltk.word_tokenize(sentence)
    stemmer = nltk.stem.snowball.FrenchStemmer()

    processed_sentence = ""

    for token in tokenized_sentence:
        token = token if not stemming else stemmer.stem(token)
        processed_sentence += token + " "

    processed_sentence = processed_sentence.strip()

    return processed_sentence if not return_tokens else process_sentence.split("")


def convert_tagged_sentence_to_bio(
    sentence: str, tag_entities_pairs: list[tuple[str, str]]
) -> str:
    """
    Given a sentence with tags, convert the sentence to BIO format.

    Args:
      sentence (str): The sentence to convert to BIO format.
      tag_entities_pairs (list[tuple[str, str]]): The tags and entities to convert to BIO format

    Returns:
      str: The sentence in BIO format

    Example:
      >>> convert_tagged_sentence_to_bio("Je voudrais voyager de <Dep>Nice<Dep> à <Arr>Clermont Ferrand<Arr>.", [("Dep", "LOC-DEP"), ("Arr", "LOC-ARR")])
      Je O
      voudrais O
      voyager O
      de O
      Nice B-LOC-DEP
      à O
      Clermont B-LOC-ARR
      Ferrand I-LOC-ARR
      . O
    """
    bare_sentence = sentence

    tags = [pair[0] for pair in tag_entities_pairs]
    entities = [pair[1] for pair in tag_entities_pairs]

    for tag in tags:
        bare_sentence = bare_sentence.replace(tag, "")

    # extended entities
    ext_entities = []
    for entity in entities:
        ext_entities.extend(["B-" + entity, "I-" + entity])

    for tag, entity in tag_entities_pairs:
        while re.search(f"{tag}(.*?){tag}", sentence):
            match = re.search(f"{tag}(.*?){tag}", sentence)
            temp_entities = [entity] * len(nltk.word_tokenize(match.group(1)))
            temp_entities[0] = "B-" + entity
            if len(temp_entities) > 1:
                for i in range(1, len(temp_entities)):
                    temp_entities[i] = "I-" + entity
            sentence = (
                sentence[: match.start()]
                + " ".join(temp_entities)
                + sentence[match.end() :]
            )

    tokens = nltk.word_tokenize(sentence)
    bare_sentence_tokens = nltk.word_tokenize(bare_sentence)

    tokenized_entities = [
        "O" if not token in ext_entities else token for token in tokens
    ]
    bio_format = [
        " ".join([token, entity])
        for token, entity in zip(bare_sentence_tokens, tokenized_entities)
    ]

    return "\n".join(bio_format)


def from_tagged_file_to_bio_file(
    input_file: str, output_file: str, tag_entities_pairs: list[tuple[str, str]]
) -> None:
    """
    Given an input file and an output file, read the input file, convert the content to BIO format, and write the content to the output file.

    Args:
      input_file (str): The path to the input file.
      output_file (str): The path to the output file.
      tag_entities_pairs (list[tuple[str, str]]): The tags and entities to convert to BIO format.
      entities (list[str]): The entities to convert to BIO format.
    """
    with open(input_file, "r") as file:
        content = file.read()

    with open(output_file, "w") as file:
        sentences = content.split("\n")
        for sentence in tqdm(sentences):
            # skip empty lines
            if not sentence:
                continue
            bio_format = convert_tagged_sentence_to_bio(sentence, tag_entities_pairs)
            file.write(bio_format + "\n")


def from_bio_file_to_examples(file_path: str, process_sentence: bool = False) -> tuple:
    """
    Given a file path, read the file and convert the content to a tuple of sentences and their respective labels vectors.

    **Note**: We are stemming the tokens instead of lemmatizing them because stemming is faster and in our case
    we are interested in getting a response the fastest way possible.

    Args:
      file_path (str): The path to the file to read.

    Returns:
      tuple: A tuple containing the inputs and labels (inputs, labels).
    """
    stop_sentences = [".", "?", "!"]
    stemmer = nltk.stem.snowball.FrenchStemmer()

    with open(file_path, "r") as file:
        content = file.read()
    lines = content.split("\n")

    sentences = []
    labels = []

    unique_labels = set()

    # getting all the unique labels
    for line in lines:
        if (len(line.split(" "))) < 2:
            continue
        word, label = line.split(" ")
        label = (
            "-".join(label.split("-")[-2:])
            if label.startswith("B") or label.startswith("I")
            else label
        )
        unique_labels.add(label)

    unique_labels = list(unique_labels)

    # "O" has to be the first label
    unique_labels = sorted(unique_labels, key=lambda x: (x != "O", x))

    # mapping labels to ids
    unique_labels = {label: i for i, label in enumerate(unique_labels)}

    # tracking the vocabulary
    vocab = set()

    sentence_words = []
    sentence_labels = []
    for line in lines:
        if (len(line.split(" "))) < 2:
            continue
        word, label = line.split(" ")
        label = (
            "-".join(label.split("-")[-2:])
            if label.startswith("B") or label.startswith("I")
            else label
        )
        word = word if not process_sentence else stemmer.stem(word)
        label = unique_labels[label]
        sentence_words.append(word)
        sentence_labels.append(label)
        vocab.add(word)
        if word in stop_sentences:
            sentences.append(" ".join(sentence_words))
            labels.append(sentence_labels)
            sentence_words = []
            sentence_labels = []

    return (sentences, labels, vocab, unique_labels)


def from_examples_to_tf_dataset(
    inputs: tuple[list[list[int]], list[list[int]]]
) -> tf.data.Dataset:
    """
    Given a tuple of inputs and labels, convert the tuple to a TensorFlow dataset.

    Args:
      inputs (tuple[list[list[int]], list[list[int]]): A tuple containing the inputs and labels (inputs, labels).

    Returns:
      tf.data.Dataset: The TensorFlow dataset.
    """

    def gen():
        for input, label in zip(inputs[0], inputs[1]):
            yield input, label

    dataset = tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None), dtype=tf.int32),
            tf.TensorSpec(shape=(None), dtype=tf.int32),
        ),
    )

    return dataset
