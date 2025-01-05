from datasets import load_from_disk

dataset = load_from_disk("./datasets/c4-lt-filtered-4-deduplicated")

def filter_short_entries(entry):
    return len(entry["text"]) >= 100


results = dataset.filter(filter_short_entries)

results.save_to_disk("./datasets/c4-lt-filtered-5-short-entries")
results.cleanup_cache_files()