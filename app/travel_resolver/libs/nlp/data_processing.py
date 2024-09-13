import nltk, re


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
