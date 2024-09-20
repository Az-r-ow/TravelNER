import nltk, re
from tqdm import tqdm


def get_tagged_content(sentence: str, tag: str) -> str:
    """
    Extract the content between two tags in a sentence given the tag.

    Args:
      sentence (str): The sentence to extract the content from.
      tag (str): The tag to extract the content between.

    Returns:
      str: The content between the tags.

    Example:
      >>> get_tagged_content("Je voudrais voyager de <Dep>Nice<Dep> à <Arr>Clermont Ferrand<Arr>.", "<Dep>")
      "Nice"
    """
    tag_match = re.search(rf"{tag}(.*?){tag}", sentence)
    if tag_match:
        return tag_match.group(1)
    return None


def process_sentence(sentence: str, dep_token="<Dep>", arr_token="<Arr>") -> tuple:
    """
    Given a sentence, extract the departure and arrival locations and tokenize the sentence.
    Then assign labels to the tokens based on whether they are part of the departure or arrival locations.
    Finally, return the logits and labels will be returned.

    Args:
      sentence (str): The sentence to process.
      dep_token (str): The token to mark the departure location.
      arr_token (str): The token to mark the arrival location.

    Returns:
      tuple: A tuple containing the logits and labels (logits, labels).
    """
    bare_sentence = sentence.replace(dep_token, "").replace(arr_token, "")
    departure = get_tagged_content(sentence, dep_token)
    arrival = get_tagged_content(sentence, arr_token)
    tokenized_sentence = nltk.word_tokenize(bare_sentence)
    labels = []
    logits = []
    for token in tokenized_sentence:
        if token in departure:
            departure_labels = [2] * len(token)
            labels.extend(departure_labels)
        elif token in arrival:
            arrival_labels = [3] * len(token)
            labels.extend(arrival_labels)
        else:
            default_labels = [1] * len(token)
            labels.extend(default_labels)
        int_chars = [ord(char) for char in token]
        logits.extend(int_chars)

    return (logits, labels)


def convert_tagged_sentence_to_bio(
    sentence: str, tags: list[str], entities: list[str]
) -> str:
    """
    Given a sentence with tags, convert the sentence to BIO format.

    Args:
      sentence (str): The sentence to convert to BIO format.
      tags (list[str]): The tags to extract the entities from.
      entities (list[str]): The entities to convert to BIO format.

    Returns:
      str: The sentence in BIO format

    Example:
      >>> convert_tagged_sentence_to_bio("Je voudrais voyager de <Dep>Nice<Dep> à <Arr>Clermont Ferrand<Arr>.", ["<Dep>", "<Arr>"], ["LOC-DEP", "LOC-ARR"])
      Je O
      voudrais O
      voyager O
      de O
      Nice LOC-DEP
      à O
      Clermont LOC-ARR
      Ferrand LOC-ARR
      . O
    """
    bare_sentence = sentence

    # extended entities
    ext_entities = []
    for entity in entities:
        ext_entities.extend(["B-" + entity, "I-" + entity])

    for tag in tags:
        bare_sentence = bare_sentence.replace(tag, "")

    for tag, entity in zip(tags, entities):
        while re.search(f"{tag}(.*?){tag}", sentence):
            match = re.search(f"{tag}(.*?){tag}", sentence)
            temp_entities = [entity] * len(nltk.word_tokenize(match.group(1)))
            temp_entities[0] = "B-" + entity
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
    input_file: str, output_file: str, tags: list[str], entities: list[str]
) -> None:
    """
    Given an input file and an output file, read the input file, convert the content to BIO format, and write the content to the output file.

    Args:
      input_file (str): The path to the input file.
      output_file (str): The path to the output file.
      tags (list[str]): The tags to extract the entities from.
      entities (list[str]): The entities to convert to BIO format.
    """
    with open(input_file, "r") as file:
        content = file.read()

    with open(output_file, "w") as file:
        sentences = content.split("\n")
        for sentence in tqdm(sentences):
            bio_format = convert_tagged_sentence_to_bio(sentence, tags, entities)
            file.write(bio_format + "\n")


def from_bio_file_to_example(file_path: str) -> tuple:
    """
    Given a file path, read the file and convert the content to a tuple of logits and labels.

    Args:
      file_path (str): The path to the file to read.

    Returns:
      tuple: A tuple containing the logits and labels (logits, labels).
    """
    stop_sentences = [".", "?", "!"]

    with open(file_path, "r") as file:
        content = file.read()
    entities = content.split("\n")
    logits = []
    labels = []

    unique_labels = set()

    # getting all the unique labels
    for entity in entities:
        if (len(entity.split(" "))) < 2:
            continue
        word, label = entity.split(" ")
        unique_labels.add(label)

    unique_labels = list(unique_labels)

    # "O" has to be the first label
    unique_labels = sorted(unique_labels, key=lambda x: (x != "O", x))

    # mapping labels to ids
    unique_labels = {label: i + 1 for i, label in enumerate(unique_labels)}

    sentence_logits = []
    sentence_labels = []
    for entity in entities:
        if (len(entity.split(" "))) < 2:
            continue
        word, label = entity.split(" ")
        ascii_code_chars = [ord(char) for char in word]
        char_labels = [unique_labels[label]] * len(ascii_code_chars)
        sentence_logits.extend(ascii_code_chars)
        sentence_labels.extend(char_labels)
        if word in stop_sentences:
            logits.append(sentence_logits)
            labels.append(sentence_labels)
            sentence_logits = []
            sentence_labels = []

    return logits, labels
