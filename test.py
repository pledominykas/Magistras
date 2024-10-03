from datasets import load_from_disk
import pandas as pd

dataset = load_from_disk("C:/Users/domin/Desktop/Magistras/c4_lithuanian_cleaned.hf")

for i in range(10):
    print(f"Entry {i}:")
    print(dataset[i]["languages"])
    print(dataset[i]["text"])

print(len(dataset))