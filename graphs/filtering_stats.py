from datasets import load_from_disk, load_dataset

dataset_0 = load_dataset("domce20/c4-lithuanian-enhanced")['train']
dataset_1 = load_from_disk("./datasets/c4-lt-filtered-1-perplexity")['train']
dataset_2 = load_from_disk("./datasets/c4-lt-filtered-2-language")['train']
dataset_3 = load_from_disk("./datasets/c4-lt-filtered-3-bad-words")['train']
dataset_4 = load_from_disk("./datasets/c4-lt-filtered-4-deduplicated")
dataset_5 = load_from_disk("./datasets/c4-lt-filtered-5-short-entries")

def get_stats(dataset, i):
    print(f"Dataset: {i}")
    print(f"Dataset size: {len(dataset)}")
    print(f"Number of characters: {sum(len(entry['text']) for entry in dataset)}")
    print()

get_stats(dataset_0, 0)
get_stats(dataset_1, 1)
get_stats(dataset_2, 2)
get_stats(dataset_3, 3)
get_stats(dataset_4, 4)
get_stats(dataset_5, 5)
