from datasets import load_from_disk

def remove_non_lithuanian(entry):
    languages = entry["languages"]
    
    if(all(language["language"] != "Language.LITHUANIAN" for language in languages)):
        return False

    return True

#https://github.com/pemistahl/lingua-py/blob/main/accuracy-reports/lingua-high-accuracy/Lithuanian.txt
def slice_out_non_lithuanian(entry):
    languages = entry["languages"]   
    languages_to_remove = [language for language in languages if language["language"] != "Language.LITHUANIAN" and language["end_index"] - language["start_index"] > 30]

    if len(languages_to_remove) == 0:
        return entry
    
    text = ""
    languages_to_keep = [language for language in languages if language not in languages_to_remove]

    for language in languages_to_keep:
        text += entry["text"][language["start_index"]:language["end_index"]]

    return {
        "text": text,
        "languages": languages_to_keep
    }


if __name__ == "__main__":
    dataset = load_from_disk("./output-k-200-with-language-results")
    
    results = dataset.filter(remove_non_lithuanian, num_proc=32)
    results = results.map(slice_out_non_lithuanian, num_proc=32)

    results.save_to_disk("./output-k-200-cleaned")