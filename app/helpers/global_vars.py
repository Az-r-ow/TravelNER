from enum import Enum


class PROGRESS(Enum):
    ANALYZING_FILE = "Analyzing file..."
    ANALYZING_AUDIO = "Analyzing audio..."
    READING_FILE = "Reading file..."
    PROCESSING = "Processing Sentences..."


class HTML_COMPONENTS(Enum):
    NO_PROMPT = "<p>Aucun prompt renseigné</p>"


entities_label_mapping = {1: "LOC-DEP", 2: "LOC-ARR"}
