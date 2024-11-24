from datasets import load_from_disk
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
    dataset = load_from_disk("./datasets/c4-lt-1-perplexity")

    dataset_language = dataset.map(detect_language, num_proc=cpu_count())

    dataset_language.save_to_disk("./datasets/c4-lt-2-language")
    dataset_language.cleanup_cache_files()