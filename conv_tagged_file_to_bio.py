from app.travel_resolver.libs.nlp.data_processing import from_tagged_file_to_bio_file


<<<<<<< HEAD
INPUT_FILES = [
    "./data/scripting_lcs_1/1k_train_large_samples.txt",
    "./data/scripting_lcs_1/10k_train_small_samples.txt",
    "./data/scripting_lcs_1/100_eval_large_samples.txt",
    "./data/scripting_lcs_1/800_eval_small_samples.txt",
]

OUTPUT_FILES = [
    "./data/bio/fr.bio/1k_train_large_samples.bio",
    "./data/bio/fr.bio/10k_train_small_samples.bio",
    "./data/bio/fr.bio/100_eval_large_samples.bio",
    "./data/bio/fr.bio/800_eval_small_samples.bio",
]
=======
INPUT_FILE = "./data/french_text/1k_unlabeled_samples.txt"
OUTPUT_FILE = "./data/bio/fr.bio/1k_unlabeled_samples.bio"
>>>>>>> e25bd1c (data: added multiple sample files + unlabeled sentences)

tag_entities_pairs = [("<Dep>", "LOC-DEP"), ("<Arr>", "LOC-ARR")]

for i, input_file in enumerate(INPUT_FILES):
    from_tagged_file_to_bio_file(input_file, OUTPUT_FILES[i], tag_entities_pairs)
