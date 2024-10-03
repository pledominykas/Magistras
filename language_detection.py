from datasets import load_from_disk, concatenate_datasets
from lingua import LanguageDetectorBuilder
from multiprocessing import cpu_count

detector = LanguageDetectorBuilder.from_all_languages().with_minimum_relative_distance(0.9).build()

def detect_language(entry):
    results = detector.detect_multiple_languages_of(entry["text"])
 
    return {
        "languages": list(map(lambda x: {
            "language": str(x.language),
            "start_index": x.start_index,
            "end_index": x.end_index,
            "word_count": x.word_count,
        }, results)),
    }

if __name__ == "__main__":
    dataset = load_from_disk("C:/Users/domin/Desktop/Magistras/c4_lithuanian.hf")
    dataset_train = dataset["train"]
    dataset_validation = dataset["validation"]
    dataset_combined = concatenate_datasets([dataset_train, dataset_validation])
    # dataset_combined = dataset_combined.take(1000)

    num_cores = cpu_count()

    dataset_combined = dataset_combined.map(detect_language, num_proc=num_cores)

    dataset_combined.save_to_disk("C:/Users/domin/Desktop/Magistras/c4_lithuanian_with_language_results.hf")