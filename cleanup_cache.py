from datasets import load_from_disk

dataset = load_from_disk("C:/Users/domin/Desktop/Magistras/c4_lithuanian_cleaned.hf")

dataset.cleanup_cache_files()