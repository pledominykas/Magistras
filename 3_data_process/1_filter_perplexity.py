from datasets import load_dataset
from multiprocessing import cpu_count
import pandas as pd

def filter_by_perplexity(entry, threshold):
    return entry["perplexity_wiki"] < threshold

if __name__ == "__main__":
    dataset = load_dataset("domce20/c4-lithuanian-enhanced")
    df = pd.DataFrame(dataset["train"]["perplexity_wiki"], columns=["perplexity"])

    threshold = df["perplexity"].quantile(0.75)

    results = dataset.filter(filter_by_perplexity, threshold=threshold, num_proc=cpu_count())

    results.save_to_disk("./datasets/c4-lt-filtered-1-perplexity")
    results.cleanup_cache_files()
