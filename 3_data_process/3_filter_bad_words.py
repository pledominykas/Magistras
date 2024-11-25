from datasets import load_from_disk
from multiprocessing import cpu_count

bad_words = []
with open('./bad_words.txt', encoding="utf8") as f:
    bad_words = f.read().splitlines()


dataset = load_from_disk("./datasets/c4-lt-filtered-1-language")

def filter_bad_words(entry):
    for bad_word in bad_words:
        if " " + bad_word + " " in entry["text"]:
            return False

    return True

results = dataset.filter(filter_bad_words, num_proc=cpu_count())

results.save_to_disk("./datasets/c4-lt-filtered-2-bad-words")