from datasets import load_dataset

dataset_wiki = load_dataset("wikimedia/wikipedia", "20231101.lt")
dataset_lrt = load_dataset("allenai/c4", "lt")

dataset_wiki = dataset_wiki["train"]
dataset_lrt = dataset_lrt["train"]

dataset_lrt = dataset_lrt.filter(lambda entry: entry["url"].startswith("https://www.lrt.lt/"))

text_file_wiki = '../datasets/lithuanian_lrt.txt'
text_file_lrt = '../datasets/lithuanian_wiki.txt'

with open(text_file_wiki, 'w', encoding='utf-8') as f:
    for example in dataset_wiki:
        text = example['text'].replace('\n', ' ')
        f.write(text + '\n')

with open(text_file_lrt, 'w', encoding='utf-8') as f:
    for example in dataset_lrt:
        text = example['text'].replace('\n', ' ')
        f.write(text + '\n')