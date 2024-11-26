from datasets import load_from_disk, load_dataset
import pandas as pd

dataset_validation = load_dataset("allenai/c4", "lt", split="validation")
dataset = load_from_disk("./datasets/c4-lt-filtered-5-short-entries")
df = pd.DataFrame(dataset["perplexity_wiki"], columns=["perplexity"])

one_percent_treshold = int(len(df) * 0.01)
five_percent_treshold = int(len(df) * 0.05)

treshold_25 = df["perplexity"].quantile(0.25)
treshold_50 = df["perplexity"].quantile(0.5)

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

# Split dataset according to perplexity quantile and shuffle
dataset_25 = dataset.filter(lambda x: x["perplexity_quantile"] == "25").shuffle()
dataset_50 = dataset.filter(lambda x: x["perplexity_quantile"] == "50").shuffle()
dataset_100 = dataset.filter(lambda x: x["perplexity_quantile"] == "100").shuffle()

# Take 1% of 25th quantile, add remaining to 50th quantile
dataset_selected_1 = dataset_25.take(one_percent_treshold)

dataset_unselected_1 = dataset_25.skip(one_percent_treshold)
dataset_50 = dataset_50.concatenate(dataset_unselected_1)

# Take 5% of 50th quantile, add remaining to 100th quantile
dataset_selected_5 = dataset_50.take(five_percent_treshold)

dataset_unselected_5 = dataset_50.skip(five_percent_treshold)
dataset_100 = dataset_100.concatenate(dataset_unselected_5)


# Save train datasets
dataset_selected_1.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-selected-1")
dataset_selected_5.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-selected-5")
dataset_100.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-100")

dataset_selected_1.cleanup_cache_files()
dataset_selected_5.cleanup_cache_files()
dataset_100.cleanup_cache_files()

# Save validation dataset
dataset_validation.save_to_disk("./datasets/c4-lt-filtered-6-perplexity-validation")
dataset_validation.cleanup_cache_files()
