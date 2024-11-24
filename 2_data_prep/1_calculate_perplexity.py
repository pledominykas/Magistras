import kenlm
from datasets import load_dataset
from multiprocessing import cpu_count

model_lrt = kenlm.Model("./models/lithuanian_lrt.binary")
model_wiki = kenlm.Model("./models/lithuanian_wikipedia.binary")

def calculate_perplexity(entry):
    text = entry["text"].replace("\n", " ")
    return {
        "perplexity_lrt": model_lrt.perplexity(text),
        "perplexity_wiki": model_wiki.perplexity(text),
    }

if __name__ == "__main__":
    dataset = load_dataset("allenai/c4", "lt")

    dataset_perplexity = dataset.map(calculate_perplexity, num_proc=cpu_count())
    dataset_perplexity.save_to_disk("../datasets/c4-lt-1-perplexity")

    dataset_perplexity.cleanup_cache_files()