from datasets import load_from_disk, load_dataset

dataset_validation = load_dataset("allenai/c4", "lt", split="validation")["validation"]
dataset = load_from_disk("./datasets/c4-lt-filtered-5-short-entries")
df = dataset.to_pandas()

treshold_1 = df["perplexity_wiki"].quantile(0.01)
treshold_5 = df["perplexity_wiki"].quantile(0.05)

def split_by_perplexity(entry):
    if entry["perplexity_wiki"] < treshold_1:
        return {
            "perplexity_quantile": "1"
        }
    elif entry["perplexity_wiki"] < treshold_5:
        return {
            "perplexity_quantile": "5"
        }
    else:
        return {
            "perplexity_quantile": "100"
        }
    
dataset = dataset.map(split_by_perplexity)

dataset_1 = dataset.filter(lambda x: x["perplexity_quantile"] == "1")
dataset_5 = dataset.filter(lambda x: x["perplexity_quantile"] == "5")
dataset_100 = dataset.filter(lambda x: x["perplexity_quantile"] == "100")

dataset_1.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-1")
dataset_5.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-5")
dataset_100.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-100")
dataset_validation.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-validation")

dataset_1.cleanup_cache_files()
dataset_5.cleanup_cache_files()
dataset_100.cleanup_cache_files()
dataset_validation.cleanup_cache_files()
