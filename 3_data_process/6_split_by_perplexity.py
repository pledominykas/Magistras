from datasets import load_from_disk, load_dataset

dataset_validation = load_dataset("allenai/c4", "lt", split="validation")["validation"]
dataset = load_from_disk("./datasets/c4-lt-filtered-5-short-entries")
df = dataset.to_pandas()

one_percent = int(len(df) * 0.01)
five_percent = int(len(df) * 0.05)
treshold_25 = df["perplexity_wiki"].quantile(0.25)
treshold_50 = df["perplexity_wiki"].quantile(0.5)

def split_by_perplexity(entry):
    if entry["perplexity_wiki"] < treshold_25:
        return {
            "perplexity_quantile": "25"
        }
    elif entry["perplexity_wiki"] < treshold_50:
        return {
            "perplexity_quantile": "50"
        }
    else:
        return {
            "perplexity_quantile": "100"
        }
    
dataset = dataset.map(split_by_perplexity)

dataset_25 = dataset.filter(lambda x: x["perplexity_quantile"] == "25")
dataset_50 = dataset.filter(lambda x: x["perplexity_quantile"] == "50")
dataset_100 = dataset.filter(lambda x: x["perplexity_quantile"] == "100")

dataset_25.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-1")
dataset_50.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-5")
dataset_100.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-100")
dataset_validation.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-validation")

dataset_25.cleanup_cache_files()
dataset_50.cleanup_cache_files()
dataset_100.cleanup_cache_files()
dataset_validation.cleanup_cache_files()
