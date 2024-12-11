import os

os.environ["TOKENIZERS_PARALLELISM"] = "1"

from transformers import AutoTokenizer
from datasets import load_dataset

raw_datasets = load_dataset("allenai/c4", "lt")

raw_datasets = raw_datasets.remove_columns(["timestamp", "url"])

def get_training_corpus():
    return (
        raw_datasets["train"][i : i + 10000]["text"]
        for i in range(0, len(raw_datasets["train"]), 10000)
    )

training_corpus = get_training_corpus()

old_tokenizer = AutoTokenizer.from_pretrained("ai-forever/mGPT")

print(os.environ["TOKENIZERS_PARALLELISM"])

tokenizer = old_tokenizer.train_new_from_iterator(training_corpus, 100000)

tokenizer.save_pretrained("mGPT-lithuanian-tokenizer")