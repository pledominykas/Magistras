#!/bin/bash
#SBATCH -p main
#SBATCH -n36

. "$HOME/.cargo/env"

python3 -m text_dedup.suffix_array --path "./datasets/c4-lt-filtered-3-bad-words" --local --k 500 --cache_dir "./cache" --output "./datasets/c4-lt-filtered-4-deduplicated" --column "text" --google_repo_path "Magistras/deduplicate-image/deduplicate-text-datasets"