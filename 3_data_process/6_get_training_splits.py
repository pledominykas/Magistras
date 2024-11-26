from datasets import load_from_disk, load_dataset

dataset_validation = load_dataset("allenai/c4", "lt", split="validation")
dataset = load_from_disk("./datasets/c4-lt-filtered-5-short-entries")

one_percent_treshold = int(len(dataset) * 0.01)
five_percent_treshold = int(len(dataset) * 0.05)

dataset = dataset.shuffle()

# Select 1% of the dataset
dataset_1 = dataset.take(one_percent_treshold)

# Select 5% of the dataset not including the 1% already selected
dataset_5 = dataset.skip(one_percent_treshold).take(five_percent_treshold)

# Select the rest of the dataset
dataset_100 = dataset.skip(one_percent_treshold + five_percent_treshold)

# Save train datasets
dataset_1.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-1")
dataset_5.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-5")
dataset_100.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-100")

dataset_1.cleanup_cache_files()
dataset_5.cleanup_cache_files()
dataset_100.cleanup_cache_files()

# Save validation dataset
dataset_validation.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-validation")
dataset_validation.cleanup_cache_files()
