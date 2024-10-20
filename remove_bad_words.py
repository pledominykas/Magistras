from datasets import load_from_disk


bad_words = []
with open('./bad_words.txt', encoding="utf8") as f:
    bad_words = f.read().splitlines()


dataset = load_from_disk("./output-k-200-cleaned")

def filter_bad_words(entry):
    for bad_word in bad_words:
        if bad_word in entry["text"]:
            return False

    return True

results = dataset.filter(filter_bad_words, num_proc=32)

results.save_to_disk("./output-k-200-cleaned-no-bad-words")