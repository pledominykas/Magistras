from datasets import load_from_disk
import pandas as pd

dataset = load_from_disk("C:/Users/domin/Desktop/Magistras/datasets/c4_lithuanian_cleaned.hf")

for i in range(len(dataset)):
    if("nahui" in dataset[i]["text"]):
        print(dataset[i]["text"])
        break