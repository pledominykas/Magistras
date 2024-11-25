from datasets import load_dataset
from multiprocessing import cpu_count

def filter_by_perplexity(entry, threshold):
    return entry["perplexity_lrt"] < threshold

if __name__ == "__main__":
    dataset = load_dataset("domce20/c4-lithuanian-enhanced")
    df = dataset.to_pandas()

    threshold = df["perplexity_lrt"].quantile(0.75) #pabandyk su wiki, ir su kitais quantile, patikrink ilgius likusio dataset

    results = dataset.filter(filter_by_perplexity, threshold=threshold, num_proc=cpu_count())

    results.save_to_disk("./datasets/c4-lt-filtered-1-perplexity")
    results.cleanup_cache_files()
    print(len(results))

